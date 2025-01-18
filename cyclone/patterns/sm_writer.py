"""
This module provides classes to write data to shared memory using Cyclone DDS.

Classes:
    SMBufferWriteField: Manages a numpy array backed by shared memory for writing.
    SMWriter: Writes shared memory buffers using DDS for synchronization.
"""

import atexit
import time
from multiprocessing import shared_memory
from typing import Dict, List

import numpy as np
from cyclonedds.domain import DomainParticipant

from cantrips.configs import load_config
from cantrips.debugging.terminal import pyout
from cyclone.idl.defaults.buffer_nr import BufferNrSample
from cyclone.idl_shared_memory.base_idl import BaseIDL
from cyclone.patterns.ddswriter import DDSWriter


class SMBufferWriteField:
    """
    Manages a numpy array backed by shared memory for writing.

    This class creates a shared memory segment and a numpy array
    that uses the shared memory as its buffer for writing data.
    """

    def __init__(self, name, shape, dtype, nbytes):
        """
        Initialize the shared memory buffer field.

        Args:
            name (str): The name of the shared memory block.
            shape (tuple): The shape of the numpy array.
            dtype (numpy.dtype): The data type of the numpy array.
            nbytes (int): The number of bytes in the shared memory buffer.
        """
        # Create a new shared memory block with the given name and size
        try:
            self.shm = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=nbytes,
            )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(
                name=name,
                create=False,
                size=nbytes,
            )
        # Create a numpy array that uses the shared memory buffer
        self.shared_array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)

        # Ensure the shared memory is properly cleaned up when the program exits
        atexit.register(self.stop)

    def stop(self):
        """Close and unlink the shared memory segment."""
        self.shm.close()
        self.shm.unlink()


class SMWriter:
    """
    Writes shared memory buffers using Cyclone DDS for synchronization.

    This class writes data to shared memory and publishes buffer numbers
    using DDS to synchronize readers.
    """

    def __init__(
        self,
        domain_participant: DomainParticipant,
        topic_name: str,
        idl_dataclass: BaseIDL,
    ):
        """
        Initialize the shared memory writer.

        Args:
            domain_participant (DomainParticipant): The DDS domain participant.
            topic_name (str): The base name of the topic.
            idl_dataclass (BaseIDL): The template defining the buffer structure.
        """
        self.config = load_config()
        self.domain_participant = domain_participant
        self.topic_name = topic_name
        self.buffer_template = idl_dataclass

        # Create a DDS writer for buffer numbers
        self.buffer_nr_writer = DDSWriter(
            domain_participant=domain_participant,
            topic_name=f"{topic_name}.buffer_nr",
            idl_dataclass=BufferNrSample,
        )

        # Create shared memory buffers
        self.buffers: List[Dict[str, SMBufferWriteField]] = self.__make_shared_memory()
        self.buffer_idx = 0

    def __call__(self, msg: BaseIDL):
        """
        Write data to shared memory and publish buffer number via DDS.

        Args:
            msg (BaseIDL): The data to write to shared memory.
        """
        # Rotate to the next buffer index
        self.buffer_idx = (self.buffer_idx + 1) % self.config.nr_of_buffers

        buffer = self.buffers[self.buffer_idx]
        # Write each field to the shared memory buffer
        for key, bufferfield in buffer.items():
            # Copy data from the message to the shared memory array
            bufferfield.shared_array[:] = getattr(msg, key)[:]

        # Publish the buffer number and timestamp via DDS
        self.buffer_nr_writer(BufferNrSample(timestamp=time.time(), nr=self.buffer_idx))

    def __make_shared_memory(self):
        """
        Create shared memory buffers based on the buffer template.

        Returns:
            List[Dict[str, SMBufferWriteField]]: A list of dictionaries, each containing buffer fields.
        """
        # Initialize a list to hold buffers for each buffer index
        buffers = [{} for _ in range(self.config.nr_of_buffers)]

        # Iterate over each field defined in the buffer template
        for name, shape, dtype, nbytes in self.buffer_template.get_fields():
            # For each buffer index, create a shared memory field
            for buffer_idx in range(self.config.nr_of_buffers):
                buffers[buffer_idx][name] = SMBufferWriteField(
                    f"{self.topic_name}.{name}.buffer_{buffer_idx}",
                    shape,
                    dtype,
                    nbytes,
                )

        return buffers
