/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

std::default_random_engine(gen);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  num_particles = 50;
  
  //Add random Gaussian Noise
  std::normal_distribution <double> dist_x (x, std[0]);
  std::normal_distribution <double> dist_y (y, std[1]);
  std::normal_distribution <double> dist_theta (theta, std[2]);
  
  for (int i = 0; i < num_particles; ++i) {
    
    Particle p;
    p.id     = i;
    p.x      = dist_x(gen);
    p.y      = dist_y(gen);
    p.theta  = dist_theta(gen);
    p.weight = 1.0; 
    
    particles.push_back(p);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::normal_distribution <double> noise_x (0.0, std_pos[0]);
  std::normal_distribution <double> noise_y (0.0, std_pos[1]);
  std::normal_distribution <double> noise_theta (0.0, std_pos[2]);
   
  for (int i = 0; i < num_particles; ++i) {
    
    double p_x     = particles[i].x;
    double p_y     = particles[i].y;
    double p_theta = particles[i].theta;
    
    //For turning scenarios
    if (abs(yaw_rate) > 1e-5) {
      
      p_x     += ((velocity/yaw_rate) * (sin(p_theta + (yaw_rate * delta_t)) - sin(p_theta)));
      p_y     += ((velocity/yaw_rate) * (cos(p_theta) - cos(p_theta + (yaw_rate * delta_t))));
      p_theta += (yaw_rate * delta_t); 
    }
    
    //For non-turn scenarios
    else {
      
      p_x     += (velocity * delta_t * cos(p_theta));
      p_y     += (velocity * delta_t * sin(p_theta));
      p_theta += (yaw_rate * delta_t);
    }
    
    //Add noise
    p_x     += noise_x(gen);
    p_y     += noise_y(gen);
    p_theta += noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (size_t i = 0; i < observations.size(); ++i) {
    
    double d_min = std::numeric_limits <double>::max();
    
    for (size_t j = 0; j < predicted.size(); ++j) {
      
      double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      
      if (d < d_min) {
        
        observations[i].id = predicted[j].id;
        d_min = d;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];

  for (int i = 0 ; i < num_particles; ++i) {
    
    double p_x     = particles[i].x; 
    double p_y     = particles[i].y;
    double p_theta = particles[i].theta;
    
    std::vector <LandmarkObs> inRange_landmarks;
    std::vector <LandmarkObs> transformed_observations;
    
    for (size_t j = 0; j < observations.size(); ++j) {
      
      LandmarkObs l;
      
//       l.id = observations[j].id;
      l.x = p_x + (cos(p_theta) * observations[j].x) - (sin(p_theta) * observations[j].y);
      l.y = p_y + (sin(p_theta) * observations[j].x) + (cos(p_theta) * observations[j].y);
      
      transformed_observations.push_back(l);
    }
    
    for (size_t j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      
      double l_id = map_landmarks.landmark_list[j].id_i;
      double l_x  = map_landmarks.landmark_list[j].x_f;
      double l_y  = map_landmarks.landmark_list[j].y_f;
      
      double d = dist(p_x, p_y, l_x, l_y);
      
      if (d < sensor_range) {
        LandmarkObs inRange_lm;
        
        inRange_lm.id = l_id;
        inRange_lm.x  = l_x;
        inRange_lm.y  = l_y;
        
        inRange_landmarks.push_back(inRange_lm);
      }
    }
    
    dataAssociation(inRange_landmarks, transformed_observations);
 
    double p_weight = 1.0;
    double mu_x, mu_y;
  
    for (size_t i = 0; i < transformed_observations.size(); ++i) {
    
      double obs_x = transformed_observations[i].x;
      double obs_y = transformed_observations[i].y;
    
      for (size_t j = 0; j < inRange_landmarks.size(); ++j) {
      
        if (transformed_observations[i].id == inRange_landmarks[j].id) {
          mu_x = inRange_landmarks[j].x;
          mu_y = inRange_landmarks[j].y;
        }
     }
    
      double prob_norm = 1/(2 * M_PI * sigma_x * sigma_y);
      double exponent  = (pow(obs_x - mu_x, 2) / (2 * pow(sigma_x, 2))) + 
                         (pow(obs_y - mu_y, 2) / (2 * pow(sigma_y, 2)));
      
      double prob = prob_norm * exp(-exponent);
      p_weight *= prob; 
  }
    particles[i].weight = p_weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::vector <double> p_weights;
  
  for (size_t i = 0; i < particles.size(); ++i) {
    
    p_weights.push_back(particles[i].weight); 
  }
  
  std::discrete_distribution <int> weighted_dist(p_weights.begin(), p_weights.end());
  std::default_random_engine gen;
  
  std::vector <Particle> resampled_particles;
  
  while (resampled_particles.size() < particles.size()) {
    
    int k = weighted_dist(gen);
    resampled_particles.push_back(particles[k]);
  }
  
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}