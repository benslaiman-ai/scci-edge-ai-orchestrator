# 🧠 SCCI Edge AI Router

**Distributed AI routing system inspired by telecom network architectures**

---

## 🚀 Overview

SCCI Edge AI Router is an experimental distributed inference system designed to orchestrate multiple AI models across heterogeneous devices such as:

* 📱 Mobile phones (edge nodes)
* 🖥️ CPUs
* ⚡ GPUs

The system dynamically selects models and execution paths based on **task complexity, context size, and system state**, using routing strategies inspired by telecom concepts like **handover, load balancing, and resilience**.

---

## 🚧 Project Status

**Active Research / Work in Progress**

This repository reflects the current state of the router.

Some features are experimental or partially implemented.
Others (such as advanced chunking) are planned but **not yet fully integrated or validated**.

---

## ⚡ Current Capabilities

### 🔀 Multi-Node Routing

* Routing across phone, CPU, and GPU nodes
* Node-aware model selection
* Fallback and failover strategies

### 🧠 Task-Aware Routing

* Classification of requests (simple / medium / complex)
* Dedicated routing for:

  * coding tasks
  * chat tasks
  * tool/function calls
  * UI helper tasks

### 📡 Telecom-Inspired Routing Logic

* Context-based escalation (handover-like behavior)
* Anti-thrashing mechanisms
* Sticky routing (session continuity)
* Predictive context scaling

### ⚙️ FastAPI Router Core

* OpenAI-compatible API
* Multi-model exposure
* Router-mode dispatching
* Internal scoring and selection logic

### 🧪 Structured Memory System (v8+)

* Importance-based memory storage
* Decay + reinforcement retrieval
* Memory poisoning protection
* Access-based reinforcement

### 🔧 Specialized Subsystems

* UI Helper isolation (lightweight tasks)
* Function/tool routing lane
* Coding lane with complexity classifier
* Long-text handling (non-chunked mode)

---

## 🏗️ Code Structure

The current implementation is **monolithic**, designed to allow rapid iteration and experimentation.

A modular architecture is planned, including:

* separated routing engine
* chunking pipeline
* scoring and validation modules
* node configuration (e.g. `nodes.yaml`)
* clearer service boundaries

---

## 🧠 Core Design Principles

### 🔹 Edge-First AI

Leverage multiple small and mid-sized models instead of relying on a single large model.

### 🔹 Adaptive Routing

Routing decisions consider:

* task complexity
* context size
* node availability
* system load

### 🔹 Telecom-Inspired Intelligence

Inspired by real-world telecom systems:

* handover strategies (context escalation)
* distributed routing
* fault tolerance
* dynamic load balancing

---

## 🏗️ Architecture (Simplified)

User Request
→ FastAPI Router
→ Task Classification
→ Routing Decision
→ Node Selection
→ Model Execution
→ Response

---

## ⚠️ Important Note on Chunking

A distributed chunking pipeline is **planned and under development**, but:

> ❗ It is **not yet fully integrated nor validated in this version**

Future versions will include:

* semantic chunking
* intelligent boundary detection
* parallel summarization pipelines

---

## 🔬 Roadmap

### v12 (in progress)

* Semantic chunking (embedding-aware)
* Model scoring system
* Improved validation logic

### v13 (planned)

* Real embeddings integration (MiniLM / E5 / BGE)
* Memory-aware routing
* Multi-model validation / voting
* Confidence scoring
* Context-aware model handover

### Future Directions

* Streaming partial results
* Cost-aware routing
* Edge-only inference pipelines
* Self-optimizing routing (learning from past tasks)

---

## 📡 About the Project

This is an **independent research project** based on practical experience in:

* telecom systems (handover, routing, resilience)
* distributed architectures
* edge computing constraints

More experiments and demos:
👉 https://benslaiman.com/

---

## 🎥 Demonstrations

Demonstrations (handover logic, routing behavior, image generation, etc.) are available via:

👉 https://benslaiman.com/lab.html

(External video demonstrations may be added later.)

---

## ⚠️ Disclaimer

This project is experimental and not production-ready.

It is intended for:

* research
* prototyping
* architectural exploration

---

## 🤝 Contributing

Feedback, ideas, and discussions are welcome.

---

## ⭐ Vision

To explore a new generation of AI systems where:

> intelligence is not centralized —
> it is **distributed, adaptive, and edge-native**
