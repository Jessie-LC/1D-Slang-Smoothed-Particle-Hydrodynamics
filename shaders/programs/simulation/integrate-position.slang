import aperture;
import include.particles.buffer;
import include.particles.constants;
import include.boundary.toroidal;
import include.kernels.normal;

extern static const int PARTICLE_COUNT;
extern static const int PARTICLE_GROUP_SIZE;

struct ComputeInput {
    uint groupIndex : SV_GroupIndex;
    uint3 groupID : SV_GroupID;
    uint3 localID : SV_GroupThreadID;
    uint3 globalID : SV_DispatchThreadID;
};

__generic<T : __BuiltinFloatingPointType, let N : int>
vector<T, N> mod(vector<T, N> x, vector<T, N> y) {
    __intrinsic_asm "mod";
}

[[shader("compute")]]
[numthreads(PARTICLE_GROUP_SIZE, 1, 1)]
void IntegratePosition(ComputeInput input) {
    uint32_t particleID = input.globalID.x;

    particle_t particle = particles[particleID];

    particle.position = particle.position + particle.velocity * deltaTime;

    float32_t tmpPosition = particle.position;
    tmpPosition = tmpPosition + boundLength / 2.0;
    tmpPosition = mod(tmpPosition, boundLength);
    tmpPosition = tmpPosition - boundLength / 2.0;
    particle.position = tmpPosition;

    particles[particleID] = particle;
}