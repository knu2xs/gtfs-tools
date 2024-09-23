GTFS-Tools API
***************

The foundation for GTFS-Tools is the :py:class:`gtfs_tools.gtfs.GtfsDataset` object. The properties representing
each of the GTFS tables represented is a subclass of :py:class:`gtfs_tools.gtfs.GtfsFile`. Of these, likely the most
useful are :py:class:`gtfs_tools.gtfs.GtfsStops` and :py:class:`gtfs_tools.gtfs.GtfsRoutes`.

.. automodule:: gtfs_tools
    :members:

gtfs_tools.gtfs
================

.. autoclass:: gtfs_tools.gtfs.GtfsDataset
    :members:

.. autoclass:: gtfs_tools.gtfs.GtfsFile
    :members:

.. autoclass:: gtfs_tools.gtfs.GtfsAgency
    :members:
    :inherited-members:

.. autoclass:: gtfs_tools.gtfs.GtfsCalendar
    :members:
    :inherited-members:

.. autoclass:: gtfs_tools.gtfs.GtfsCalendarDates
    :members:
    :inherited-members:

.. autoclass:: gtfs_tools.gtfs.GtfsFrequencies
    :members:
    :inherited-members:

.. autoclass:: gtfs_tools.gtfs.GtfsRoutes
    :members:
    :inherited-members:

.. autoclass:: gtfs_tools.gtfs.GtfsShapes
    :members:
    :inherited-members:

.. autoclass:: gtfs_tools.gtfs.GtfsStopTimes
    :members:
    :inherited-members:

.. autoclass:: gtfs_tools.gtfs.GtfsStops
    :members:
    :inherited-members:

.. autoclass:: gtfs_tools.gtfs.GtfsTrips
    :members:
    :inherited-members:

gtfs_tools.utils
=================

Utilities included to speed up the development and creation of build scripts.

.. automodule:: gtfs_tools.utils
    :members:

gtfs_tools.utils.gtfs
======================

Utility functions specific to working with GTFS data.

.. automodule:: gtfs_tools.utils.gtfs
    :members:

.. _Spatially enabled data frame: https://developers.arcgis.com/python/api-reference/arcgis.features.toc.html#geoaccessor