public module NormalKernel;

float32_t NormalKernel(in float32_t x, in float32_t sLength) {
    float32_t standardDeviation = sLength * 0.25f;
    float32_t norm = (1.0f / (standardDeviation * sqrt(2.0f * 3.14159f)));
    return norm * exp(-0.5f * pow(x / standardDeviation, 2.0f));
}

float32_t NormalKernel_Derivative(in float32_t x, in float32_t sLength) {
    float32_t standardDeviation = sLength * 0.25f;
    float32_t norm = (1.0f / (standardDeviation * sqrt(2.0f * 3.14159f)));
    return norm * -((sqrt(pow(x, 2.0f)) * exp(-0.5f * pow(x/standardDeviation, 2.0f))) / pow(standardDeviation, 2.0f));
}

float32_t NormalKernel_DerivativeRadius(in float32_t x, in float32_t sLength) {
    float32_t standardDeviation = sLength * 0.25f;
    float32_t norm = (1.0f / (standardDeviation * sqrt(2.0f * 3.14159f)));
    return norm * ((pow(x, 2.0f) * exp(-0.5f * pow(x/standardDeviation, 2.0f))) / pow(standardDeviation, 3.0f));
}
