#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>

#include "particle_filter.h"

using namespace std;

std::default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	//   Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++)
	{
		Particle PARTICLE;
		PARTICLE.id = i;
		PARTICLE.x = dist_x(gen);
		PARTICLE.y = dist_y(gen);
		PARTICLE.theta = dist_theta(gen);
		PARTICLE.weight = 1;

		particles.push_back(PARTICLE);
		weights.push_back(1);
	}
	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// define normal distributions for sensor noise

  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    // calculate new state
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//   Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  for(auto& obs: observations){
		// init minimum distance to a high number
    double min_dist = std::numeric_limits<float>::max();

    for(const auto& pred: predicted){
			// find the predicted landmark nearest the current observed landmark
      double distance = dist(obs.x, obs.y, pred.x, pred.y);
      if( min_dist > distance){
        min_dist = distance;
        obs.id = pred.id;
      }
    }
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks){
		//   Update the weights of each particle using a mult-variate Gaussian distribution. More information:
		//   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
		// NOTE: The observations are given in the VEHICLE'S coordinate system. The particles are located
		//   according to the MAP'S coordinate system so a transformation is needed.
		//   The transformation requires both rotation AND translation (but no scaling).
		//   Used theory: https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
		//   Resource for the actual equation to implement (equation 3.33)
		//   http://planning.cs.uiuc.edu/node99.html

  for(auto& PARTICLE: particles){
    PARTICLE.weight = 1.0;

    // collect landmarks within sensor range
    vector<LandmarkObs> predictions;
    for(const auto& lm: map_landmarks.landmark_list){
      double distance = dist(PARTICLE.x, PARTICLE.y, lm.x_f, lm.y_f);
      if( distance < sensor_range){ // if the landmark is within the sensor range, save it to predictions
        predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }

    // Transformation from vehicle to map coordinates
    vector<LandmarkObs> observations_map;
    double cos_theta = cos(PARTICLE.theta); // pre-calculation for efficiency
    double sin_theta = sin(PARTICLE.theta); // pre-calculation for efficiency

    for(const auto& obs: observations){
      LandmarkObs trans_obs;
      trans_obs.x = obs.x * cos_theta - obs.y * sin_theta + PARTICLE.x;
      trans_obs.y = obs.x * sin_theta + obs.y * cos_theta + PARTICLE.y;
      trans_obs.id = obs.id;
      observations_map.push_back(trans_obs);
    }

		// Using dataAssociation for finding landmark index and associating predicted with observed landmarks
    dataAssociation(predictions, observations_map);

		// Update weights
    for(const auto& obs_m: observations_map){

      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);
			// Calculate Multi-variate Gaussian
      double x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      double y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double MV_Gaussian = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      PARTICLE.weight *=  MV_Gaussian;
    }
    weights.push_back(PARTICLE.weight);
  }
}


void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // generate distribution according to weights
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  // create resampled particles
  vector<Particle> resampled_particles;
  resampled_particles.resize(num_particles);

  // resample the particles in accordance to the weights
  for(int i=0; i<num_particles; i++){
    int idx = dist(gen);
    resampled_particles[i] = particles[idx];
  }
  particles = resampled_particles;
  weights.clear();
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
