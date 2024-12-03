# Shared Memory Data Exchange with Cyclone DDS

This repository provides a set of Python classes for efficiently sharing data between processes using shared memory, with synchronization handled by [Cyclone DDS](https://projects.eclipse.org/projects/iot.cyclonedds). It is designed for high-performance applications where large numpy arrays need to be shared between a single writer and multiple readers.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Usage](#usage)
  - [Defining a Shared Memory IDL](#defining-a-buffer-template)
  - [Setting Up the Writer](#setting-up-the-writer)
  - [Setting Up the Reader](#setting-up-the-reader)
- [Limitations](#limitations)

## Overview

The provided classes allow for the sharing of numpy arrays between processes without the overhead of serialization. By using shared memory, large data structures can be accessed by multiple processes as if they were local, eliminating the need for copying data between processes.

Cyclone DDS is used for synchronization between the writer and readers. The writer notifies the readers when new data is available by publishing buffer numbers via DDS topics.

## Key Features

- **Efficient Data Sharing**: Uses shared memory to share numpy arrays between processes, avoiding data copying.
- **DDS Synchronization**: Utilizes Cyclone DDS for synchronizing access to shared buffers.
- **Customizable Buffers**: Allows defining custom data structures using numpy arrays.
- **Single Writer, Multiple Readers**: Designed for one writer and multiple readers.

## Architecture

The system consists of the following components:

- **BufferTemplate**: A base class for defining the structure of shared data using dataclasses and numpy arrays.
- **Shared Memory Writer (SMWriter)**: Manages writing data to shared memory and publishing buffer numbers via DDS.
- **Shared Memory Reader (SMReader)**: Manages reading data from shared memory and synchronizing with the writer via DDS.
- **Buffer Fields**: Individual shared memory segments representing fields of the data structure.

## Usage

### Defining a Buffer Template

First, define a buffer template by extending the `BufferTemplate` class. This template specifies the data fields to be shared and their shapes and data types.

```python
from dataclasses import dataclass
import numpy as np
from cyclone.idl_shared_memory.base_idl import BaseIDL


@dataclass
class FrameBuffer(BaseIDL):
  # Timestamp of the frame
  timestamp: np.ndarray = np.empty((1,), dtype=np.float64)
  # Color image data (height x width x channels)
  color: np.ndarray = np.empty((480, 848, 3), dtype=np.uint8)
  # Depth image data (height x width)
  depth: np.ndarray = np.empty((480, 848), dtype=np.uint16)
  # Extrinsic camera parameters (transformation matrix)
  extrinsics: np.ndarray = np.empty((4, 4), dtype=np.float64)
  # Intrinsic camera parameters (camera matrix)
  intrinsics: np.ndarray = np.empty((3, 3), dtype=np.float64)
```

### Setting Up the Writer

Create an instance of `SMWriter` using your buffer template. The writer will handle writing data to shared memory and notifying readers via DDS.

```python
from cyclonedds.domain import DomainParticipant
from cyclone.patterns.sm_writer import SMWriter

# Initialize the DDS domain participant
domain_participant = DomainParticipant()

# Create the shared memory writer
writer = SMWriter(
    domain_participant=domain_participant,
    topic_name="frame_data",
    idl_dataclass=FrameBuffer()
)

# Write data to shared memory
frame_data = FrameBuffer(
    timestamp=np.array([time.time()], dtype=np.float64),
    color=your_color_array,
    depth=your_depth_array,
    extrinsics=your_extrinsics_matrix,
    intrinsics=your_intrinsics_matrix
)

writer(frame_data)
```

### Setting Up the Reader

Create an instance of `SMReader` to read data from shared memory. The reader listens to DDS topics for buffer numbers to know when new data is available.

```python
from cyclonedds.domain import DomainParticipant
from cyclone.patterns.sm_reader import SMReader

# Initialize the DDS domain participant
domain_participant = DomainParticipant()

# Create the shared memory reader
reader = SMReader(
    domain_participant=domain_participant,
    topic_name="frame_data",
    idl_dataclass=FrameBuffer()
)

# Read data from shared memory
try:
    while True:
        frame_data = reader()
        process_frame_data(frame_data)
except WaitingForFirstMessageException:
    print("Waiting for data...")
```

## Limitations

- **Numpy Arrays Only**: The system currently supports only numpy arrays as data fields.
- **Known Shapes and Data Types**: The shape and `dtype` of each numpy array must be known in advance and consistent between the writer and readers.
- **Single Writer**: Only one writer process is supported to prevent race conditions and ensure data consistency.