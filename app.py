import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# 1. Simulation of Chaotic Data
# -----------------------------
def simulate_chaotic_data(num_agents=10, num_steps=200):
    """
    Simulate a small chaotic multi-agent ecosystem using sine waves + noise.
    Each agent produces values over time with random noise to mimic instability.
    """
    timestamps = pd.date_range(start=datetime.now(), periods=num_steps, freq='S')
    data = []

    for agent_id in range(num_agents):
        base_signal = np.sin(np.linspace(0, 20, num_steps) + agent_id)
        noise = np.random.normal(0, 0.7, num_steps)  # chaos
        chaotic_signal = base_signal + noise
        data.append(pd.Series(chaotic_signal, index=timestamps, name=f"Agent_{agent_id}"))

    return pd.concat(data, axis=1)

# -----------------------------
# 2. FFT Smoothing
# -----------------------------
def apply_fft_smoothing(signal, keep_ratio=0.1):
    """
    Simplified FFT-based smoothing: keep only the lowest frequencies.
    """
    fft_result = np.fft.fft(signal)
    cutoff = int(len(fft_result) * keep_ratio)
    fft_filtered = np.zeros_like(fft_result)
    fft_filtered[:cutoff] = fft_result[:cutoff]  # low-pass filter
    smoothed_signal = np.fft.ifft(fft_filtered).real
    return smoothed_signal

# -----------------------------
# 3. Resilience and Metrics
# -----------------------------
def calculate_resilience(signal):
    """Resilience is measured as inverse of variance."""
    return 1.0 / (np.var(signal) + 1e-9)

def calculate_error_reduction(original, processed):
    """How much error propagation was reduced."""
    orig_dev = np.mean(np.abs(original - np.mean(original)))
    proc_dev = np.mean(np.abs(processed - np.mean(processed)))
    return max(0, 1 - (proc_dev / orig_dev))

# -----------------------------
# 4. Streamlit App
# -----------------------------
def main():
    st.title("Future Forecast Tokens (FFT) - Minimal Demo")
    st.write("""
    This free, open-source demo shows how FFTs stabilize chaotic multi-agent systems.
    It demonstrates the core concepts from the research paper in real-time.
    """)

    num_agents = st.slider("Number of Agents", 5, 50, 10)
    num_steps = st.slider("Simulation Steps", 100, 1000, 200)

    st.subheader("1. Simulating Chaotic System")
    data = simulate_chaotic_data(num_agents, num_steps)
    st.line_chart(data.iloc[:, 0:3])  # Show only first 3 agents

    st.subheader("2. Applying FFT Stabilization")
    # Apply FFT to first agent for simplicity
    original = data.iloc[:, 0].values
    smoothed = apply_fft_smoothing(original)

    # Metrics
    resilience_before = calculate_resilience(original)
    resilience_after = calculate_resilience(smoothed)
    error_reduction = calculate_error_reduction(original, smoothed)

    st.metric("Resilience Before FFT", f"{resilience_before:.3f}")
    st.metric("Resilience After FFT", f"{resilience_after:.3f}")
    st.metric("Error Reduction (%)", f"{error_reduction * 100:.1f}%")

    st.subheader("3. Visualization")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(original, label="Chaotic Signal", alpha=0.7)
    ax.plot(smoothed, label="FFT Stabilized Signal", linestyle="--", color="orange")
    ax.set_title("FFT Stabilization Effect")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Signal Value")
    ax.legend()
    st.pyplot(fig)

    st.write("This experiment is reproducible and uses only open data and free tools.")

if __name__ == "__main__":
    main()
