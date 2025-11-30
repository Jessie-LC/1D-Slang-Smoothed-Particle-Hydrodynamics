import aperture;
import include.particles.buffer;
import include.particles.constants;
import include.boundary.toroidal;
import include.kernels.normal;
import include.physics.constants;

extern static const int PARTICLE_COUNT;
extern static const int PARTICLE_GROUP_SIZE;

struct ComputeInput {
    uint groupIndex : SV_GroupIndex;
    uint3 groupID : SV_GroupID;
    uint3 localID : SV_GroupThreadID;
    uint3 globalID : SV_DispatchThreadID;
};

precise float32_t IdealGasLaw(float32_t density) {
    return density * R_ideal * 273.0f;
}

static const float32_t alpha = 0.002f;
static const float32_t beta = alpha * 2.0f;

[[shader("compute")]]
[numthreads(PARTICLE_GROUP_SIZE, 1, 1)]
void CalculateAcceleration(ComputeInput input) {
    uint32_t particleID = input.globalID.x;

    particle_t particle = particles[particleID];
    precise float32_t acceleration = 0.0f;
    for(int32_t i = 0; i < PARTICLE_COUNT; ++i) {
        if (i == particleID) {
            continue;
        }

        particle_t found = particles[i];

        precise float32_t dist = ToroidalDistance(found.position, particle.position);
        precise float32_t dx = ToroidalDifference(found.position, particle.position);
        precise float32_t dv = found.velocity - particle.velocity;
        precise float32_t direction = dx / dist;
        if (dist == 0.0f) {
            direction = 0.0f;
        }
        precise float32_t slope = NormalKernel_Derivative(dist, smoothingLength);

        precise float32_t viscPiA = 0.0f;
        precise float32_t viscPiB = 0.0f;
        {
            precise float32_t muA = (smoothingLength * dot(dx, dv)) / (pow(dist, 2.0) + 1e-5 * pow(smoothingLength, 2.0));
            precise float32_t muB = (smoothingLength * dot(dx, dv)) / (pow(dist, 2.0) + 1e-5 * pow(smoothingLength, 2.0));
            viscPiA = (-alpha * 331.0f * muA + beta * muA) / particle.density;
            viscPiB = (-alpha * 331.0f * muB + beta * muB) / found.density;
        }

        precise float32_t pressureA = IdealGasLaw(particle.density) / (particle.density*particle.density);
        precise float32_t pressureB = IdealGasLaw(found.density) / (found.density*found.density);
        precise float32_t pressureAcceleration = (pressureA * slope * direction) + (pressureB * slope * direction);
        precise float32_t viscousAcceleration = (particle.mass * viscPiA * direction * slope) + (found.mass * viscPiB * direction * slope);
        acceleration += (found.mass * pressureAcceleration) + viscousAcceleration;
    }

    particle.acceleration = acceleration;

    particles[particleID] = particle;
}