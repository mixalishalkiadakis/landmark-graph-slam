#pragma once

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Rot2.h>

#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/linear/NoiseModel.h>

#include <unordered_map>
#include <vector>
#include <cstddef>

class SlamBackend {
    public:
        struct LandmarkEstimate {
            std::size_t id;
            gtsam::Point2 position;
        };

        SlamBackend();

        void initialize(const gtsam::Pose2& initial_pose);

        void addOdometry (
            const gtsam::Pose2& relative_motion, 
            double dt
        );

        void addLandmarkObservation(
            std::size_t landmark_id,
            const gtsam::Rot2& bearing,
            double range
        );

        void update();

        gtsam::Pose2 currentPose() const;

        std::vector<LandmarkEstimate> landmarks() const;

        std::vector<gtsam::Pose2> trajectory() const;

    private:

        gtsam::ISAM2 isam_;

        gtsam::NonlinearFactorGraph new_factors_;

        gtsam::Values new_values_;

        std::size_t current_pose_index_;

        bool initialized_;

        std::unordered_map<std::size_t, gtsam::Symbol> landmark_keys_;

        gtsam::SharedNoiseModel prior_noise_;
        gtsam::SharedNoiseModel odometry_noise_;
        gtsam::SharedNoiseModel observation_noise_;

};