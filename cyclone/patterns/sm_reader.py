"""
This module provides classes to read data from shared memory using Cyclone DDS.

Classes:
    SharedMemoryNoResourceTracker: A wrapper around SharedMemory that unregisters from the resource tracker.
    SMBufferReadField: Manages a numpy array backed by shared memory.
    SMReader: Reads shared memory buffers using DDS for synchronization.
"""

import atexit
import time
from multiprocessing import shared_memory, resource_tracker

import numpy as np
from cyclonedds.domain import DomainParticipant

from cantrips.configs import load_config
from cantrips.exceptions import WaitingForFirstMessageException
from cantrips.logging.logger import get_logger
from cyclone.idl.defaults.buffer_nr import BufferNrSample
from cyclone.idl_shared_memory.base_idl import BaseIDL
from cyclone.patterns.ddsreader import DDSReader


class SharedMemoryNoResourceTracker:
    """
    Wrapper around shared_memory.SharedMemory that unregisters from the resource tracker.

    This prevents the resource_tracker from automatically unlinking the shared memory segment
    when the process exits, which is useful when multiple processes need to share the memory.
    """
    def __init__(self, name):
        """
        Initialize the shared memory segment.

        Args:
            name (str): The name of the existing shared memory block to attach to.
        """
        self.shm = shared_memory.SharedMemory(name=name)
        # Unregister from resource_tracker to prevent automatic unlinking
        resource_tracker.unregister("/" + self.shm.name, "shared_memory")

    def close(self):
        """Close the shared memory segment."""
        self.shm.close()

    @property
    def buf(self):
        """Return the buffer interface to the shared memory."""
        return self.shm.buf


class SMBufferReadField:
    """
    Manages a numpy array backed by shared memory for reading.

    This class attaches to a shared memory segment and creates a numpy array
    that uses the shared memory as its buffer.
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
        self.shm = SharedMemoryNoResourceTracker(name=name)
        self.shared_array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)

        # Ensure the shared memory is properly closed when the program exits
        atexit.register(self.stop)

    def stop(self):
        """Close the shared memory segment."""
        self.shm.close()


logger = get_logger()


class SMReader:
    """
    Reads shared memory buffers using Cyclone DDS for synchronization.

    This class reads data from shared memory that is synchronized using DDS.
    It listens to a DDS topic for buffer numbers and reads the corresponding
    buffers from shared memory.
    """
    def __init__(
        self,
        domain_participant: DomainParticipant,
        topic_name: str,
        idl_dataclass: BaseIDL,
    ):
        """
        Initialize the shared memory reader.

        Args:
            domain_participant (DomainParticipant): The DDS domain participant.
            topic_name (str): The base name of the topic.
            idl_dataclass (BaseIDL): The template defining the buffer structure.
        """
        self.config = load_config()

        self.domain_participant = domain_participant
        self.topic_name = topic_name
        self.buffer_template = idl_dataclass

        # Create a DDS reader for buffer numbers
        self.buffer_nr_reader = DDSReader(
            domain_participant=domain_participant,
            topic_name=f"{topic_name}.buffer_nr",
            idl_dataclass=BufferNrSample,
        )
        # Wait for the writer to start publishing
        self.__wait_for_writer()

        # Load the shared memory buffers
        self.buffers = self.__load_shared_memory()

    def __call__(self):
        """
        Read the latest data from shared memory.

        Returns:
            An instance of buffer_template.__class__ containing the data.

        Raises:
            WaitingForFirstMessageException: If no data is available yet.
        """
        buffer_nr_sample = self.buffer_nr_reader()
        if buffer_nr_sample is None:
            raise WaitingForFirstMessageException
        buffer = self.buffers[buffer_nr_sample.nr]

        kwargs = {}
        for key, bufferfield in buffer.items():
            # Copy the data from shared memory into a numpy array
            kwargs[key] = bufferfield.shared_array.copy()

        return self.buffer_template.__class__(**kwargs)

    def __load_shared_memory(self):
        """
        Load shared memory buffers based on the buffer template.

        Returns:
            list: A list of dictionaries, each containing buffer fields.
        """
        # Initialize a list to hold buffers for each buffer index
        buffers = [{} for _ in range(self.config.nr_of_buffers)]

        # Iterate over each field defined in the buffer template
        for name, shape, dtype, nbytes in self.buffer_template.get_fields():
            # For each buffer index, create a shared memory field
            for buffer_idx in range(self.config.nr_of_buffers):
                buffers[buffer_idx][name] = SMBufferReadField(
                    f"{self.topic_name}.{name}.buffer_{buffer_idx}",
                    shape,
                    dtype,
                    nbytes,
                )
        return buffers

    def __wait_for_writer(self, timeout=None, warn_every=60):
        """
        Wait for the DDS writer to start publishing buffer numbers.

        Args:
            timeout (float, optional): The maximum time to wait in seconds. If None, wait indefinitely.
            warn_every (int): The interval in seconds to log a warning message.

        Raises:
            RuntimeError: If the writer does not start within the timeout period.
        """
        t0 = time.time()
        warned = 0
        while self.buffer_nr_reader() is None and (
            timeout is None or (time.time() - t0) < timeout
        ):
            if warn_every * (warned + 1) < (time.time() - t0):
                warned += 1

                # Construct a warning message about waiting time
                warning_msg = (
                    f"Shared Memory Reader {self.topic_name} has been "
                    f"waiting for Shared Memory Writer for more than "
                    f"{warned*warn_every} seconds. "
                )
                if timeout is None:
                    warning_msg += f"No timeout defined; will wait indefinitely."
                else:
                    warning_msg += (
                        f"Timeout: {timeout} seconds. Will wait for "
                        f"{timeout-warned*warn_every} more seconds."
                    )

                logger.warning(warning_msg)

            # Sleep briefly to yield control and prevent busy waiting
            self.domain_participant.sleep()

        if self.buffer_nr_reader() is None:
            # If the writer has not started within the timeout, raise an error
            raise RuntimeError(
                f"Shared Memory Reader {self.topic_name} timed out "
                f"waiting for buffer."
            )
