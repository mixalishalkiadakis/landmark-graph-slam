#include "slam_backend.hpp"


#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/linear/LossFunctions.h>


#include <stdexcept>
#include <cmath>

namespace {

gtsam::ISAM2Params makeISAMParams() {
    gtsam::ISAM2Params params;

    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;

    return params;
}

}

SlamBackend::SlamBackend() 
    : 
    isam_(makeISAMParams()),
    current_pose_index_(0),  
    initialized_(false) 
{
    prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.01, 0.01, 0.01));
    odometry_noise_ = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.02, 0.02, 0.01));

    const auto gaussian_observation_noise = gtsam::noiseModel::Diagonal::Sigmas(
        gtsam::Vector2(0.1, 0.05)
    );

    observation_noise_ = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(2.5),
        gaussian_observation_noise
    );
}

void SlamBackend::initialize(const gtsam::Pose2& initial_pose) {
    
    if (initialized_) {
        throw std::logic_error("Backend is initialized");
    }
    
    gtsam::Symbol x0('x', 0);
    new_factors_.addPrior(x0, initial_pose, prior_noise_);

    new_values_.insert(x0, initial_pose);
    isam_.update(new_factors_, new_values_);

    new_factors_.resize(0);
    new_values_.clear();

    current_pose_index_ = 0;
    initialized_ = true;
}

void SlamBackend::addOdometry(const gtsam::Pose2& relative_motion, double dt) {
    
    if (!initialized_) {
        throw std::logic_error("Backend is not initialized");
    }

    if(!std::isfinite(dt) || dt <= 0) {
        throw std::invalid_argument("Odometry dt must be finite and positive");
    }

    constexpr double reference_dt = 0.1;
    const double scale = std::sqrt(dt / reference_dt);

    const auto motion_noise = gtsam::noiseModel::Diagonal::Sigmas(
        gtsam::Vector3(
            0.02 * scale,
            0.02 * scale,
            0.01 * scale
        )
    );
    
    gtsam::Symbol prev_symbol('x', current_pose_index_);
    current_pose_index_ ++;
    gtsam::Symbol new_symbol('x', current_pose_index_);

    new_factors_.add(gtsam::BetweenFactor<gtsam::Pose2>(
        prev_symbol,
        new_symbol,
        relative_motion,
        motion_noise
    ));

    gtsam::Pose2 previous_pose;

    if(new_values_.exists(prev_symbol)) {
        previous_pose = new_values_.at<gtsam::Pose2>(prev_symbol);
    }
    else {
        previous_pose = isam_.calculateEstimate<gtsam::Pose2>(prev_symbol);
    }

    new_values_.insert(new_symbol, previous_pose.compose(relative_motion));

}

void SlamBackend::addLandmarkObservation(
    std::size_t landmark_id,
    const gtsam::Rot2& bearing,
    double range
) {
    
    if (!initialized_) {
        throw std::logic_error("Backend is not initialized");
    }

    if (!std::isfinite(range) || range <= 0.0) {
        throw std::invalid_argument(
            "Landmark range must be finite and positive"
        );
    }
    
    double range_sigma;

    if(range < 2.0) {
        range_sigma = 0.08;
    }
    else if (range < 3.0) {
        range_sigma = 0.12;
    }
    else {
        range_sigma = 0.20;
    }

    constexpr double bearing_sigma = 0.1;

    const auto gaussian_noise = gtsam::noiseModel::Diagonal::Sigmas(
        gtsam::Vector2(bearing_sigma, range_sigma)
    );

    const auto measurement_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(2.5),
        gaussian_noise
    );

    gtsam::Symbol curr_pose('x', current_pose_index_);
    gtsam::Symbol curr_landmark('l', landmark_id);
    

    //if new landmark, then we have to calculate initial estimate
    if(landmark_keys_.find(landmark_id) == landmark_keys_.end()) {
        landmark_keys_[landmark_id] = curr_landmark;
        gtsam::Pose2 current_pose;

        //if observation isnt made on a new pose
        if(new_values_.exists(curr_pose)) {
            current_pose = new_values_.at<gtsam::Pose2>(curr_pose);
        }
        else{
            current_pose = isam_.calculateEstimate<gtsam::Pose2>(curr_pose);
        }

        //calculate the global estimate for the landmark
        gtsam::Point2 landmark_estimate = current_pose.transformFrom(
            gtsam::Point2 (range * bearing.c(), range*bearing.s())
        ); 
        
        new_values_.insert(curr_landmark, landmark_estimate);
    }

    new_factors_.add(gtsam::BearingRangeFactor<gtsam::Pose2,gtsam::Point2>(
            curr_pose,
            curr_landmark,
            bearing,
            range,
            measurement_noise
        ));
}

void SlamBackend::update() {
    
    if (!initialized_) {
        throw std::logic_error("Backend is not initialized");
    }
    
    isam_.update(new_factors_, new_values_);
    new_factors_.resize(0);
    new_values_.clear();
}

gtsam::Pose2 SlamBackend::currentPose() const {
    
    if (!initialized_) {
        throw std::logic_error("Backend is not initialized");
    }
    
    gtsam::Symbol current_symbol('x', current_pose_index_);
    return isam_.calculateEstimate<gtsam::Pose2>(current_symbol);
}

std::vector<SlamBackend::LandmarkEstimate> SlamBackend::landmarks() const {
    
    if (!initialized_) {
        throw std::logic_error("Backend is not initialized");
    }
    
    std::vector<SlamBackend::LandmarkEstimate> landmark_vector;
    for(const auto& landmark: landmark_keys_) {
        landmark_vector.push_back({
            landmark.first,
            isam_.calculateEstimate<gtsam::Point2>(landmark.second)
            }
        );
    }

    return landmark_vector;
}

std::vector<gtsam::Pose2> SlamBackend::trajectory() const {
    if (!initialized_) {
        throw std::logic_error("Backend is not initialized");
    }

    const auto estimate = isam_.calculateEstimate();

    std::vector<gtsam::Pose2> poses;
    poses.reserve(current_pose_index_ + 1);

    for (std::size_t i = 0; i <= current_pose_index_; ++i) {

        poses.push_back(
            estimate.at<gtsam::Pose2>(gtsam::Symbol('x', i))
        );

    }

    return poses;
}