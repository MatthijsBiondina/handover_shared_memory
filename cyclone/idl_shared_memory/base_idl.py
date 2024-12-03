"""
This module defines the BufferTemplate class, which serves as a base class for buffer templates using dataclasses.

Classes:
    BufferTemplate: Base class that provides a method to retrieve numpy array field information.
"""

from dataclasses import dataclass, fields
import numpy as np

@dataclass
class BaseIDL:
    """
    Base class for buffer templates using dataclasses.

    Provides a method to retrieve information about numpy array fields,
    which is useful for setting up shared memory buffers for inter-process communication.
    """

    def get_fields(self):
        """
        Generator that yields information about the numpy array fields.

        Iterates over all fields defined in the dataclass and yields a tuple
        containing the field name, shape, dtype, and number of bytes for each numpy array field.

        Yields:
            tuple: (field_name, shape, dtype, nbytes) of each numpy array field.
        """
        for field in fields(self):
            array = getattr(self, field.name)
            # Only process fields that are numpy arrays
            if isinstance(array, np.ndarray):
                yield (field.name, array.shape, array.dtype, array.nbytes)
