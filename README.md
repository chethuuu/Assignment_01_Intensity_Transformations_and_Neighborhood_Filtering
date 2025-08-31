content = """# Assignment 01 — Piecewise Intensity Transform (LUT)

## Introduction

This assignment demonstrates a simple **intensity transformation** on a grayscale image using a **piecewise linear lookup table (LUT)**. The goal is to enhance mid-tones while keeping very dark and very bright regions mostly unchanged, following the provided input–output plot.

## What we are doing

### 01
- Convert the input image to **grayscale**.
- Build a **256-entry LUT** that maps each input intensity `x` (0–255) to an output `y` using three segments:
  - **Segment A (0 ≤ x ≤ 50):** `y = x` (keep darkest tones unchanged)
  - **Segment B (50 < x ≤ 150):** linear boost from `(50, 100)` to `(150, 255)` with slope ≈ **1.55** (enhance mid-tones)
  - **Segment C (150 < x ≤ 255):** `y = x` (keep brightest tones unchanged)
    > Note: In this implementation, `x = 150` is included in Segment B, so `y(150) = 255`.
- Apply the LUT to every pixel (**output = LUT[input]**) and **save** the transformed result.
