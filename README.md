# Greenova: Energy-Adaptive AI for Onboard Satellite Inference
 
**IASTAM 6.0 Technical Challenge — Track 1, Problem 2 (TUNSA Collaboration)**
 
## Context
 
Satellites do not have a stable power supply: every orbit alternates between a sunlight phase (~60 min, high energy) and an eclipse phase (~30 min, declining energy). Onboard AI must keep working in both cases, without causing mission failure.
 
## The Greenova Mechanism
 
Greenova treats energy not as a fixed constraint, but as a dynamic variable that drives the AI model's behavior in real time — the "Energy-Inference Pendulum":
 
| Energy State | Model Behavior |
|---|---|
| High (State 1) | Full network deployed — maximum accuracy |
| Nominal (State 2) | Intermediate layers bypassed — balanced accuracy/speed |
| Critical (State 3) | Shallowest model, reduced inference frequency — core system survival |
 
Implemented using a MobileNetV2 fine-tuned on EuroSAT (satellite land-cover classification, 10 classes), fitted with 3 early exits corresponding to the 3 states, and an orbital simulator that automatically switches between them based on the simulated energy level.
 
## Key Results — Model Compression
 
Two quantization approaches were tested to reduce the size of each state:
 
| Method | Size Reduction | Accuracy Loss |
|---|---|---|
| Naive static quantization (PTQ) | up to 71% | up to -20 points (unacceptable) |
| **Quantization-Aware Training (QAT)** | **54% to 71%** | **-1.6 to -3.2 points only** |
 
Naive PTQ significantly degrades accuracy on MobileNetV2, a phenomenon documented on depthwise-separable architectures. QAT achieves the same level of compression while preserving near-intact accuracy — making compression viable for real onboard deployment.
 
### QAT Results per Energy State
 
| State   | Acc (fine-tuned) | Acc (QAT) | Size Reduction |
|---------|-------------------|-----------|-----------------|
| Deep    | 93.60%            | 91.00%    | 71.1%           |
| Mid     | 95.20%            | 93.60%    | 66.7%           |
| Shallow | 78.80%            | 75.60%    | 53.6%           |
 
## Tech Stack
 
- Python, PyTorch, TorchGeo
- Dataset: [EuroSAT](https://github.com/phelber/EuroSAT) (10 land-use classes)
## Repository Structure
 
- `greenova_notebook.ipynb` — full notebook: EuroSAT data loading, MobileNetV2 fine-tuning, Greenova multi-exit mechanism, orbital energy simulator, PTQ vs QAT quantization
## Next Steps (Phase 3)
 
- Integrate the routing mechanism with the live orbital energy simulator for full end-to-end automated switching during inference
- Measure inference latency and estimated energy consumption per exit
- Stress-test the system under irregular power profiles beyond the idealized sunlight/eclipse cycle
## Team
 
Team Greenova — ESSTHS Student Branch Chapter, IEEE Industry Applications Society
IASTAM 6.0 Technical Challenge / TUNSA Collaboration
