#include "slam_backend.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include <variant>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Rot2.h>

#include <iomanip>
#include <locale>
#include <algorithm>
#include <cmath>

struct OdometryRecord {
    double timestamp;
    double forward_velocity;
    double angular_velocity;
};

struct RawMeasurement {
    double timestamp;
    int barcode;
    double range;
    double bearing;
};

struct LandmarkObservation
{
    double timestamp;
    std::size_t landmark_id;
    double range;
    double bearing;
};

struct RelativeMotion {
    double dx;
    double dy;
    double dtheta;
    double dt;
    double timestamp;
};

struct OdometrySample {
    double timestamp;
    gtsam::Pose2 pose;
};

std::unordered_map<int, int> loadBarcodeMap() {

    std::unordered_map<int, int> barcode_to_subject;

    //loading Barcodes.bat
    std::ifstream file("dataset/Barcodes.dat");
    if (!file.is_open()) {std::cerr << "Could not open file\n";
    return{};}

    //parsing
    std::string line;
    while(std::getline(file,line)) {
        if(line.empty() || line[0] == '#') {continue;}
        std::istringstream iss(line);
        int subject;
        int barcode;
        if (iss >> subject >> barcode ) {
            barcode_to_subject[barcode] = subject;
        }
    }

    return barcode_to_subject;

}

std::vector<OdometryRecord> loadOdometry() {
    std::vector<OdometryRecord> odometry;

    //loading Robot1_Odometry.bat
    std::ifstream file("dataset/Robot1_Odometry.dat");
    if (!file.is_open()) {std::cerr << "Could not open file\n"; return{};}

    //parsing
    std::string line;
    while(std::getline(file,line)) {
        std::istringstream iss(line);
        OdometryRecord odom;
        if (
            iss 
            >> odom.timestamp
            >> odom.forward_velocity
            >> odom.angular_velocity 

        ) {
            odometry.push_back(odom);
        }
    }

    return odometry;
}

std::vector<RawMeasurement> loadMeasurements() {
    std::vector<RawMeasurement> measurements;

    //loading Robot1_Odometry.bat
    std::ifstream file("dataset/Robot1_Measurement.dat");
    if (!file.is_open()) {std::cerr << "Could not open file\n"; return{};}

    //parsing
    std::string line;
    while(std::getline(file,line)) {
        std::istringstream iss(line);
        RawMeasurement raw;
        if (
            iss 
            >> raw.timestamp
            >> raw.barcode
            >> raw.range
            >> raw.bearing
        ) {
            measurements.push_back(raw);
        }
    }

    return measurements;
}

std::vector<LandmarkObservation> convertLandmarks (
    const std::vector<RawMeasurement>& measurements,
    const std::unordered_map<int, int>& barcode_to_subject 
)
{

    std::vector<LandmarkObservation> landmarks;
    for (const auto& measurement: measurements) {
        
        auto subject_id = barcode_to_subject.find(measurement.barcode);

        if(subject_id != barcode_to_subject.end()) {
            if(subject_id->second >= 6 && subject_id->second <= 20) {
                LandmarkObservation landmark;
                landmark.timestamp = measurement.timestamp;
                landmark.landmark_id = subject_id->second;
                landmark.range = measurement.range;
                landmark.bearing = measurement.bearing;
                landmarks.push_back(landmark);
            }
        }
    }

    return landmarks;
}

RelativeMotion integrateMotion (
    double forward_velocity,
    double angular_velocity,
    double start_time,
    double end_time
)
{
    RelativeMotion motion;
    double dt = end_time - start_time;

    if (dt <= 0.0) {
        throw std::runtime_error("Odometry timestamps are out of order");
    }

    motion.dt = dt;
    //for small angular velocity we assume the vehicle follows a straight line
    if(std::abs(angular_velocity) < 0.01) {
        motion.dx = forward_velocity * dt;
        motion.dy = 0;
        motion.dtheta = angular_velocity * dt;
    }
    //else we use the unicycle model
    else {
        const double radius =
        forward_velocity / angular_velocity;
        const double angle = angular_velocity * dt;
        motion.dx = radius * std::sin(angle);
        motion.dy = radius * (1.0 - std::cos(angle));
        motion.dtheta = angle;
    }
    motion.timestamp = end_time;

    return motion;   
}


std::vector<OdometrySample> ProcessDataset(
    SlamBackend& backend,
    const std::vector<OdometryRecord>& odometry,
    const std::vector<LandmarkObservation>& observations,
    double window_frame,
    double odometry_window
)
{

    //--------------------------------------------------------------------
    // Safety checks
    //--------------------------------------------------------------------
    if (odometry.size() < 2) {
        throw std::runtime_error(
            "At least two odometry records are required"
        );
    }

    if (observations.empty()) {
        throw std::runtime_error(
            "No landmark observations were loaded"
        );
    }

    if (window_frame <= 0.0) {
        throw std::runtime_error(
            "Frame window must be positive"
        );
    }

    if (!std::isfinite(odometry_window) || odometry_window <= 0.0) {
        throw std::runtime_error(
            "Odometry window must be finite and positive"
        );
    }

    //--------------------------------------------------------------------
    // indexing and initialization
    //--------------------------------------------------------------------

    std::size_t odom_index = 0;
    std::size_t obs_index = 0;
    int frame_counter = 0;

    gtsam::Pose2 accumulated_motion(0,0,0);
    double odom_time_cursor = odometry.front().timestamp;


    //trajectory logging initialization
    gtsam::Pose2 odometry_pose(0.0, 0.0, 0.0);
    std::vector<OdometrySample> odometry_trajectory;
    odometry_trajectory.push_back({
        odometry.front().timestamp,
        odometry_pose
    });

    //all observations made in this window will be tied to one pose
    std::vector<LandmarkObservation> landmark_cashe;
    double frame_start_time = 0.0;
    double frame_end_time = 0.0;

    //--------------------------------------------------------------------
    // visualization stuff
    //--------------------------------------------------------------------

    std::ofstream online_file("online_trajectory.csv");

    if (!online_file.is_open()) {
        throw std::runtime_error("Could not create online_trajectory.csv");
    }

    online_file.imbue(std::locale::classic());
    online_file << std::setprecision(17);

    online_file
        << "pose_id,timestamp,"
        << "odom_x,odom_y,odom_theta,"
        << "slam_x,slam_y,slam_theta\n";

    auto log_online_pose = [&](std::size_t pose_id, double timestamp) {
        const auto slam_pose = backend.currentPose();

        online_file
            << pose_id << ','
            << timestamp << ','
            << odometry_pose.x() << ','
            << odometry_pose.y() << ','
            << odometry_pose.theta() << ','
            << slam_pose.x() << ','
            << slam_pose.y() << ','
            << slam_pose.theta() << '\n';

        online_file.flush();
    };

    log_online_pose(0, odometry.front().timestamp);


    //--------------------------------------------------------------------
    // MAIN LOOP
    //--------------------------------------------------------------------


    while(obs_index < observations.size()) {
        
        
        //--------------------------------------------------------------------
        //create new frame and reset motion
        //--------------------------------------------------------------------
        landmark_cashe.clear();
        accumulated_motion = gtsam::Pose2{0,0,0};
        frame_start_time = observations[obs_index].timestamp;
        frame_end_time = frame_start_time + window_frame;
        


        //at this point, odom time cursor is at the last integration time.
        //We want to check here if the time that has passed between then and the time of
        //this current observation is greater than some threshold.
        //If that is true, we add odometry only in frames until we reach the timestamp of the
        //observation

        double unaccounted_motion_start = odom_time_cursor;
        double next_pose_time = unaccounted_motion_start + odometry_window;
        while (next_pose_time < frame_start_time) {     

            if (odom_index + 1 >= odometry.size()) {
                throw std::runtime_error(
                    "Odometry ended before the next pose"
                );
            }

            const double next_odom_time = odometry[odom_index + 1].timestamp;

            const double integration_end = std::min(
                next_pose_time,
                next_odom_time
            );

            RelativeMotion relative_motion = integrateMotion(
                odometry[odom_index].forward_velocity,
                odometry[odom_index].angular_velocity,
                odom_time_cursor,
                integration_end
            );

            accumulated_motion = accumulated_motion.compose(
                gtsam::Pose2(
                    relative_motion.dx,
                    relative_motion.dy,
                    relative_motion.dtheta
                )
            );


            odom_time_cursor = integration_end; 
            if (odom_time_cursor >= next_odom_time) {
                ++odom_index;
            }

            if(odom_time_cursor >= next_pose_time) {

                const double curr_motion_dt =
                    odom_time_cursor - unaccounted_motion_start;

                backend.addOdometry(accumulated_motion, curr_motion_dt);

                odometry_pose = odometry_pose.compose(accumulated_motion);

                odometry_trajectory.push_back({
                    odom_time_cursor,
                    odometry_pose
                });

                backend.update();

                log_online_pose(frame_counter + 1, odom_time_cursor);
                ++frame_counter;

                accumulated_motion = gtsam::Pose2{0, 0, 0};
                unaccounted_motion_start = odom_time_cursor;

                next_pose_time = unaccounted_motion_start + odometry_window;

            }

        }
        
        //now a time window of next_obs_timestep - odom_time cursor is left
        //we have to integrate that motion too

        while (odom_time_cursor < frame_start_time) {
            if (odom_index + 1 >= odometry.size()) {
                throw std::runtime_error(
                    "Odometry ended before the observation"
                );
            }

            const double next_odom_time =
                odometry[odom_index + 1].timestamp;

            const double integration_end = std::min(
                frame_start_time,
                next_odom_time
            );

            const RelativeMotion relative_motion = integrateMotion(
                odometry[odom_index].forward_velocity,
                odometry[odom_index].angular_velocity,
                odom_time_cursor,
                integration_end
            );

            accumulated_motion = accumulated_motion.compose(
                gtsam::Pose2(
                    relative_motion.dx,
                    relative_motion.dy,
                    relative_motion.dtheta
                )
            );

            odom_time_cursor = integration_end;

            if (odom_time_cursor >= next_odom_time) {
                ++odom_index;
            }
        }

        const double remaining_dt =
        odom_time_cursor - unaccounted_motion_start;

        if (remaining_dt > 0.0) {
            backend.addOdometry(accumulated_motion, remaining_dt);
        
            odometry_pose = odometry_pose.compose(accumulated_motion);
        
            odometry_trajectory.push_back({
                odom_time_cursor,
                odometry_pose
            });
        
            backend.update();
        
            log_online_pose(frame_counter + 1, odom_time_cursor);
            ++frame_counter;
        }

        accumulated_motion = gtsam::Pose2{0, 0, 0};



        //--------------------------------------------------------------------
        //add observations in the time window
        //--------------------------------------------------------------------
        while(obs_index < observations.size() && 
            observations[obs_index].timestamp <= frame_end_time
        ) {
            landmark_cashe.push_back(observations[obs_index]);
            obs_index ++;
        }

        //--------------------------------------------------------------------
        //integrate motion up to window end
        //--------------------------------------------------------------------

        constexpr double time_epsilon = 1e-9;
        const double motion_start_time = odom_time_cursor;

        while (odom_time_cursor + time_epsilon < frame_end_time) {
        
            if (odom_index + 1 >= odometry.size()) {
                throw std::runtime_error(
                    "Odometry ended before the current frame"
                );
            }
        
            const double next_odom_time =
                odometry[odom_index + 1].timestamp;
            

            const double integration_end =
                std::min(frame_end_time, next_odom_time);
        
            RelativeMotion relative_motion = integrateMotion(
                odometry[odom_index].forward_velocity,
                odometry[odom_index].angular_velocity,
                odom_time_cursor,
                integration_end
            );
        
            accumulated_motion = accumulated_motion.compose(
                gtsam::Pose2(
                    relative_motion.dx,
                    relative_motion.dy,
                    relative_motion.dtheta
                )
            );
        
            odom_time_cursor = integration_end;
        
            if (
                odom_time_cursor >=
                next_odom_time - time_epsilon
            ) {
                ++odom_index;
                odom_time_cursor = next_odom_time;
            }
        }


        //log trajectory
        odometry_pose =
            odometry_pose.compose(accumulated_motion);

        odometry_trajectory.push_back({
            frame_end_time,
            odometry_pose
        });
        

        //--------------------------------------------------------------------
        //send accumulated motion to backend
        //--------------------------------------------------------------------
        const double motion_dt = odom_time_cursor - motion_start_time;
        backend.addOdometry(accumulated_motion, motion_dt);


        //--------------------------------------------------------------------
        //send landmarks to backend
        //--------------------------------------------------------------------
        
        for(const auto& landmark: landmark_cashe) {
            backend.addLandmarkObservation(
                landmark.landmark_id,
                gtsam::Rot2{landmark.bearing},
                landmark.range
            );
        }
        

        //--------------------------------------------------------------------
        //update backend
        //--------------------------------------------------------------------
        backend.update();

        log_online_pose(frame_counter + 1, frame_end_time);

        ++frame_counter;

        std::cout
            << "Frame " << frame_counter
            << " | time: " << frame_start_time
            << " -> " << frame_end_time
            << " | observations: "
            << landmark_cashe.size()
            << '\n';
    }

    std::cout
        << "Processed frames: "
        << frame_counter
        << '\n';

    std::cout
        << "Processed observations: "
        << obs_index
        << " / "
        << observations.size()
        << '\n';


    return odometry_trajectory;
}

void exportResults(
    const std::vector<OdometrySample>& odometry_trajectory,
    const SlamBackend& backend
)
{
    const auto& slam_trajectory =
        backend.trajectory();

    if (
        odometry_trajectory.size() !=
        slam_trajectory.size()
    ) {
        throw std::runtime_error(
            "Odometry and SLAM trajectory sizes do not match: "
            + std::to_string(odometry_trajectory.size())
            + " vs "
            + std::to_string(slam_trajectory.size())
        );
    }

    //----------------------------------------------------------------
    // Export trajectories
    //----------------------------------------------------------------
    std::ofstream trajectory_file("trajectory.csv");

    if (!trajectory_file.is_open()) {
        throw std::runtime_error(
            "Could not create trajectory.csv"
        );
    }

    trajectory_file.imbue(
        std::locale::classic()
    );

    trajectory_file
        << std::setprecision(17);

    trajectory_file
        << "pose_id,timestamp,"
        << "odom_x,odom_y,odom_theta,"
        << "slam_x,slam_y,slam_theta\n";

    for (
        std::size_t i = 0;
        i < slam_trajectory.size();
        ++i
    ) {
        const gtsam::Pose2& odom_pose =
            odometry_trajectory[i].pose;

        const gtsam::Pose2& slam_pose =
            slam_trajectory[i];

        trajectory_file
            << i << ','
            << odometry_trajectory[i].timestamp << ','
            << odom_pose.x() << ','
            << odom_pose.y() << ','
            << odom_pose.theta() << ','
            << slam_pose.x() << ','
            << slam_pose.y() << ','
            << slam_pose.theta()
            << '\n';
    }

    trajectory_file.close();

    //----------------------------------------------------------------
    // Export estimated landmarks
    //----------------------------------------------------------------
    const auto& estimated_landmarks =
        backend.landmarks();

    std::ofstream landmark_file("landmarks.csv");

    if (!landmark_file.is_open()) {
        throw std::runtime_error(
            "Could not create landmarks.csv"
        );
    }

    landmark_file.imbue(
        std::locale::classic()
    );

    landmark_file
        << std::setprecision(17);

    landmark_file
        << "landmark_id,x,y\n";

    for (const auto& landmark : estimated_landmarks) {
        landmark_file
            << landmark.id << ','
            << landmark.position.x() << ','
            << landmark.position.y()
            << '\n';
    }

    landmark_file.close();

    std::cout
        << "Exported "
        << slam_trajectory.size()
        << " poses to trajectory.csv\n";

    std::cout
        << "Exported "
        << estimated_landmarks.size()
        << " landmarks to landmarks.csv\n";
}

int main()
{
    try {
        //----------------------------------------------------------------
        // Load dataset
        //----------------------------------------------------------------
        const auto barcode_to_subject =
            loadBarcodeMap();

        const auto odometry =
            loadOdometry();

        const auto raw_measurements =
            loadMeasurements();

        const auto observations =
            convertLandmarks(
                raw_measurements,
                barcode_to_subject
            );

        std::cout
            << "Loaded "
            << barcode_to_subject.size()
            << " barcode mappings\n";

        std::cout
            << "Loaded "
            << odometry.size()
            << " odometry records\n";

        std::cout
            << "Loaded "
            << raw_measurements.size()
            << " raw measurements\n";

        std::cout
            << "Kept "
            << observations.size()
            << " landmark observations\n";

        //----------------------------------------------------------------
        // Initialize SLAM
        //----------------------------------------------------------------
        SlamBackend backend;

        const gtsam::Pose2 initial_pose(
            0.0,
            0.0,
            0.0
        );

        backend.initialize(initial_pose);

        //----------------------------------------------------------------
        // Process dataset using observation keyframes
        //----------------------------------------------------------------
        constexpr double window_frame = 0.01;
        constexpr double odom_frame = 0.2;

        const std::vector<OdometrySample>
            odometry_trajectory =
                ProcessDataset(
                    backend,
                    odometry,
                    observations,
                    window_frame,
                    odom_frame
                );

        //----------------------------------------------------------------
        // Export odometry, SLAM and landmarks
        //----------------------------------------------------------------
        exportResults(
            odometry_trajectory,
            backend
        );

        std::cout
            << "Dataset processing completed successfully\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << '\n';

        return 1;
    }
}