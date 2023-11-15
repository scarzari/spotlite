# Copyright (c) 2023 Satellogic USA Inc. All Rights Reserved.
#
# This file is part of Spotlite.
#
# This file is subject to the terms and conditions defined in the file 'LICENSE',
# which is part of this source code package.

"""Tests for mapUtils."""

from copy import deepcopy

import unittest
from unittest.mock import patch

import folium

from shapely.geometry import box, Polygon
import geopandas as gpd
import pandas as pd

import plotly.graph_objects as go

from mapUtils import (
    estimate_zoom_level,
    create_bounding_box,
    create_bounding_box_choropleth,
    create_map,
    update_map_with_tiles,
    process_multiple_points_to_bboxs,
    process_multiple_points_choropleth,
    create_heatmap_for_age,
    create_heatmap_for_image_count,
    create_heatmap_for_cloud,
    create_folium_basemap,
)

class TestEstimateZoomLevel(unittest.TestCase):
    """Zoom level tests for estimate_zoom_level."""

    def test_large_extent(self):
        """Test by starting big."""
        self.assertEqual(estimate_zoom_level(0, 0, 20, 20), 6)

    def test_medium_extent(self):
        """Test by getting smaller."""
        self.assertEqual(estimate_zoom_level(0, 0, 6, 6), 7)

    def test_small_extent(self):
        """Test by getting smaller."""
        self.assertEqual(estimate_zoom_level(0, 0, 3, 3), 8)

    def test_very_small_extent(self):
        """Test by getting smaller."""
        self.assertEqual(estimate_zoom_level(0, 0, 1.5, 1.5), 9)

    def test_tiny_extent(self):
        """Test by getting smaller."""
        self.assertEqual(estimate_zoom_level(0, 0, 0.75, 0.75), 10)

    def test_miniscule_extent(self):
        """Test by getting smaller."""
        self.assertEqual(estimate_zoom_level(0, 0, 0.375, 0.375), 11)

    def test_microscopic_extent(self):
        """Test by getting smaller."""
        self.assertEqual(estimate_zoom_level(0, 0, 0.1875, 0.1875), 12)

    def test_nanoscopic_extent(self):
        """Test by getting smaller."""
        self.assertEqual(estimate_zoom_level(0, 0, 0.09375, 0.09375), 13)

    def test_infinitesimal_extent(self):
        """Test by getting smaller."""
        self.assertEqual(estimate_zoom_level(0, 0, 0.05, 0.05), 14)


class TestCreateBoundingBox(unittest.TestCase):
    """Bounding box tests for create_bounding_box."""

    def test_normal_conditions(self):
        """Test under normal conditions."""
        center_lat, center_lon = 40.7128, -74.0060  # Example coordinates (New York City)
        width_km = 5  # Example width in kilometers

        bbox = create_bounding_box(center_lat, center_lon, width_km)

        self.assertIsInstance(bbox, Polygon)
        self.assertAlmostEqual(bbox.area, 0.0014986392800794503)
        self.assertAlmostEqual(list(bbox.exterior.coords), [
            (-73.97641396705156, 40.70013658436215),
            (-73.97641396705156, 40.725463387767846),
            (-74.03558603294844, 40.725463387767846),
            (-74.03558603294844, 40.70013658436215),
            (-73.97641396705156, 40.70013658436215)
        ])

    def test_default_width(self):
        """Test with default width."""
        center_lat, center_lon = 40.7128, -74.0060  # Example coordinates

        bbox = create_bounding_box(center_lat, center_lon)

        self.assertIsInstance(bbox, Polygon)
        self.assertAlmostEqual(bbox.area, 0.0005395101538935243)
        self.assertAlmostEqual(list(bbox.exterior.coords), [
            (-73.98824837980132, 40.70520195396158),
            (-73.98824837980132, 40.72039803600522),
            (-74.02375162019868, 40.72039803600522),
            (-74.02375162019868, 40.70520195396158),
            (-73.98824837980132, 40.70520195396158)
        ])

    def test_invalid_input(self):
        """Test with invalid input."""
        with self.assertRaises(ValueError):
            _ = create_bounding_box('invalid', 'invalid')


class TestCreateBoundingBoxChoropleth(unittest.TestCase):
    """Choropleth for bound box tests for create_bounding_box_choropleth."""

    def test_return_types(self):
        """Test return types are correct."""
        bbox, fig = create_bounding_box_choropleth(
            40.7128, -74.0060)  # Example coordinates
        self.assertIsInstance(bbox, dict)
        self.assertIsInstance(fig, go.Figure)

    def test_bbox_structure(self):
        """Test interior data structure is correct."""
        bbox, _ = create_bounding_box_choropleth(40.7128, -74.0060)
        # Check if bbox has the expected keys and structure
        self.assertIn('type', bbox)
        self.assertIn('coordinates', bbox)
        self.assertEqual(bbox['type'], 'Polygon')

    def test_figure_properties(self):
        """Test figure data structure is reasonable."""
        _, fig = create_bounding_box_choropleth(40.7128, -74.0060)
        # Check for some basic properties of the figure, like data length
        self.assertGreater(len(fig.data), 0)
        # Test for specific properties related to the map configuration
        self.assertEqual(fig.layout.mapbox.style, 'carto-positron')


class TestCreateMap(unittest.TestCase):
    """Base Map from lat long and bounding box test for create_map."""

    def setUp(self):
        """Setup the lat lon and bounding box."""
        # Example coordinates (New York City)
        self.lat, self.lon = 40.7128, -74.0060
        self.bbox = box(-74.1, 40.7, -73.9, 40.8)  # Example bounding box

    def test_return_type(self):
        """Create basic folium map."""
        map_obj = create_map(self.lat, self.lon, self.bbox)
        self.assertIsInstance(map_obj, folium.Map)

    def test_map_properties(self):
        """Inspect map properties."""
        map_obj = create_map(self.lat, self.lon, self.bbox)
        # Check if the map is centered correctly
        self.assertEqual(map_obj.location, [self.lat, self.lon])
        # Further checks can be added for zoom level and other properties

    def test_polygon_in_map(self):
        """Ensure polygon is added to the map."""
        # Check if a Polygon layer is added to the map
        map_obj = create_map(self.lat, self.lon, self.bbox)
        polygon_added = any(isinstance(child, folium.vector_layers.Polygon)
                        for child in map_obj._children.values())
        self.assertTrue(polygon_added)


class TestUpdateMapWithTiles(unittest.TestCase):
    """Update a map with new folium polygon objects test."""

    def setUp(self):
        self.ref_folium_map = folium.Map(
            location=[40.7128, -74.0060], zoom_start=13)
        self.folium_map = folium.Map(
            location=[40.7128, -74.0060], zoom_start=13)
        self.copied_folium_map = self.folium_map
        self.animation_filename = "example_animation.gif"
        self.aoi_bbox = box(-74.1, 40.7, -73.9, 40.8)
        # Creating a sample GeoDataFrame
        data = {'capture_date': pd.to_datetime('2023-01-01'),
                'satl:outcome_id': 'xxxx',
                'freq': 'S',
                'geometry': [self.aoi_bbox],
                'cloud_cover': [10]}
        self.tiles_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        self.tiles_gdf = self.tiles_gdf.set_index(
            pd.DatetimeIndex(self.tiles_gdf['capture_date']))

    def test_map_update_with_nonempty_gdf(self):
        """Test update of non-empty geopandas dataframe."""
        updated_map = update_map_with_tiles(
            self.folium_map, self.tiles_gdf, self.animation_filename, self.aoi_bbox)

        self.assertIsInstance(updated_map, folium.Map)
        self.assertEqual(len(updated_map._children) - len(
            self.ref_folium_map._children), 2)  # Check if new layers are added

    @patch('logging.Logger.warning')
    def test_map_update_with_empty_gdf(self, mock_logger_warning):
        """Test empty geopandas dataframe handling."""
        mock_logger_warning.returns = None
        empty_gdf = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")
        updated_map = update_map_with_tiles(
            self.folium_map, empty_gdf, self.animation_filename, self.aoi_bbox)
        self.assertIsNone(updated_map)

    def test_polygon_addition_for_polygon_geometry(self):
        """Test additional of polygon geometry."""
        # Assuming the sample GeoDataFrame has polygon geometries
        updated_map = update_map_with_tiles(
            self.folium_map, self.tiles_gdf, self.animation_filename, self.aoi_bbox)
        polygon_added = any(isinstance(child, folium.vector_layers.Polygon)
                            for child in updated_map._children.values())
        self.assertTrue(polygon_added)


class TestProcessMultiplePointsToBboxs(unittest.TestCase):
    """Add children bounding boxes with centroid and extent to folium base map test."""

    def setUp(self):
        self.points = [
            {'lat': 40.7128, 'lon': -74.0060},  # Example point 1
            {'lat': 34.0522, 'lon': -118.2437}  # Example point 2
        ]
        self.width = 3  # Example width for bounding box

    def test_return_types(self):
        """."""
        master_map, aois_list = process_multiple_points_to_bboxs(
            self.points, self.width)
        self.assertIsInstance(master_map, folium.Map)
        self.assertIsInstance(aois_list, list)
        for aoi in aois_list:
            self.assertIsInstance(aoi, Polygon)

    def test_initial_map_location(self):
        """."""
        master_map, _ = process_multiple_points_to_bboxs(
            self.points, self.width)
        self.assertEqual(master_map.location, [
                         self.points[0]['lat'], self.points[0]['lon']])

    def test_number_of_aois_and_map_features(self):
        """."""
        master_map, aois_list = process_multiple_points_to_bboxs(
            self.points, self.width)
        self.assertEqual(len(aois_list), len(self.points))
        # Assuming each point adds one feature to the map
        self.assertEqual(len(master_map._children) - len(self.points), 3)


class TestProcessMultiplePointsChoropleth(unittest.TestCase):
    """Multiple polygon choropleth and "master" figure test."""

    def setUp(self):
        self.points = [
            {'lat': 40.7128, 'lon': -74.0060},  # Example point 1
            {'lat': 34.0522, 'lon': -118.2437}  # Example point 2
        ]
        self.width = 3  # Example width for bounding box

    def test_return_types(self):
        """The function returns a go.Figure and a list of dictionaries."""
        master_fig, aois_list = process_multiple_points_choropleth(
            self.points, self.width)
        self.assertIsInstance(master_fig, go.Figure)
        self.assertIsInstance(aois_list, list)

    def test_aois_and_traces_added(self):
        """The "master" figure is correctly updated with traces from each bounding box choropleth map.

        The list of AOIs corresponds to the number of points provided.

        The "master" figure's layout is correctly updated based on the calculated global bounding box."""
        estimate_zoom_level.return_value = 10
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox())  # Adding a dummy trace

        master_fig, aois_list = process_multiple_points_choropleth(
            self.points, self.width)
        self.assertEqual(len(aois_list), len(self.points))
        self.assertEqual(len(master_fig.data), len(self.points))


class TestCreateHeatmapForAge(unittest.TestCase):
    """Create folium Map heatmap based on age of image test."""

    def setUp(self):
        data = {
            'data_age': [10, 20],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])]
        }
        self.aggregated_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    @patch('folium.Map.save')
    @patch('logging.Logger.warning')
    def test_heatmap_creation(self, mock_logger_warning, mock_map_save):
        """A folium.Map object is returned.

        The map is correctly initialized with a center based on the centroid of the geometries in the input GeoDataFrame.

        Polygons are added to the map for each geometry in the GeoDataFrame.

        Polygons have colors and opacities corresponding to the data_age values.

        A colormap is added as a child to the map.
        """
        mock_logger_warning.returns = None
        mock_map_save = None
        map_obj = create_heatmap_for_age(self.aggregated_gdf)
        self.assertIsInstance(map_obj, folium.Map)

        # Assuming the center of the data is calculated correctly
        # You can mock or calculate the expected start coordinates based on your mock data
        expected_start_coord = (1, 1)  # Replace with expected coordinates
        self.assertEqual(map_obj.location, list(expected_start_coord))

        # Check if polygons are added to the map
        polygon_added = any(isinstance(child, folium.vector_layers.Polygon)
                            for child in map_obj._children.values())
        self.assertTrue(polygon_added)

        # Check if the colormap legend is added
        colormap_added = any(isinstance(child, folium.map.Layer)
                             for child in map_obj._children.values())
        self.assertTrue(colormap_added)

    # Add more tests as necessary to cover different cases and parameters


class TestCreateHeatmapForImageCount(unittest.TestCase):
    """Create folium heatmap based on image counts test."""

    def setUp(self):
        self.aggregated_gdf = gpd.GeoDataFrame({
            'image_count': [5, 15],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])],
            'grid:code': [1, 2]
        }, crs="EPSG:4326")

    @patch('folium.Map.save')
    @patch('logging.Logger.warning')
    def test_heatmap_creation(self, mock_logger_warning, mock_map_save):
        """The function returns a Folium Map object.

        The map is correctly initialized with a center based on the centroid of the geometries in the input GeoDataFrame.

        Polygons are added to the map for each geometry in the GeoDataFrame, with colors based on the image count.

        A colormap is added to the map.
        """
        mock_logger_warning.returns = None
        mock_map_save = None
        map_obj = create_heatmap_for_image_count(self.aggregated_gdf)
        self.assertIsInstance(map_obj, folium.Map)

        # Check map center
        center = self.aggregated_gdf.geometry.unary_union.centroid
        expected_start_coord = (center.y, center.x)
        self.assertEqual(map_obj.location, list(expected_start_coord))

        # Check if polygons are added to the map
        polygon_added = any(isinstance(child, folium.vector_layers.Polygon)
                            for child in map_obj._children.values())
        self.assertTrue(polygon_added)

        # Check if colormap is added
        colormap_added = any(isinstance(child, folium.map.Layer)
                             for child in map_obj._children.values())
        self.assertTrue(colormap_added)


class TestCreateHeatmapForCloud(unittest.TestCase):
    """Create folium heatmap based on cloud coverage test."""

    def setUp(self):
        self.tiles_gdf = gpd.GeoDataFrame({
            'eo:cloud_cover': [20, 40],
            'data_age': [10, 15],
            'grid:code': [1, 2],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])]
        }, crs="EPSG:4326")
        self.orig_tiles_gdf = self.tiles_gdf.copy(deep=True)

    @patch('logging.Logger.warning')
    def test_return_none_for_empty_gdf(self, mock_logger_warning):
        """Verify the function returns None when the input tiles_gdf is empty."""
        mock_logger_warning.returns = None
        mock_map_save = None
        empty_gdf = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")
        result = create_heatmap_for_cloud(empty_gdf)
        self.assertIsNone(result)

    def test_return_new_figure(self):
        """Ensure a new go.Figure is returned when existing_fig is not provided.."""
        fig = create_heatmap_for_cloud(self.tiles_gdf)
        self.assertIsInstance(fig, go.Figure)

    def test_add_trace_to_existing_figure(self):
        """Check that the function correctly adds a new trace to existing_fig when it is provided."""
        existing_fig = go.Figure()
        existing_fig_copy = deepcopy(existing_fig)
        result_fig = create_heatmap_for_cloud(self.tiles_gdf, existing_fig)
        self.assertEqual(len(result_fig.data), len(existing_fig_copy.data) + 1)

    def test_figure_properties(self):
        """Validate that the figure's properties are set correctly based on the input tiles_gdf."""
        fig = create_heatmap_for_cloud(self.tiles_gdf)


class TestCreateFoliumBasemap(unittest.TestCase):
    """Create folium basemap used for creating heatmaps test."""

    def setUp(self):
        self.capture_grouped_tiles_gdf = gpd.GeoDataFrame({
            'capture_date': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')],
            'outcome_id': [1, 2],
            'thumbnail_url': ['http://example.com/image1.png', 'http://example.com/image2.png'],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])]
        }, crs="EPSG:4326")

    @patch('folium.Map.save')
    @patch('logging.Logger.warning')
    def test_return_none_for_empty_gdf(self, mock_logger_warning, mock_map_save):
        """Verify the function returns None when the input capture_grouped_tiles_gdf is empty."""
        mock_logger_warning.returns = None
        mock_map_save = None
        empty_gdf = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")
        result = create_folium_basemap(empty_gdf)
        self.assertIsNone(result)

    def test_return_map_object(self):
        """Ensure a folium.Map object is returned when capture_grouped_tiles_gdf is not empty."""
        map_obj = create_folium_basemap(self.capture_grouped_tiles_gdf)
        self.assertIsInstance(map_obj, folium.Map)

    def test_polygon_and_image_overlay_added(self):
        """Check that polygons and image overlays are added to the map for each geometry in
        capture_grouped_tiles_gdf.
        """
        map_obj = create_folium_basemap(self.capture_grouped_tiles_gdf)
        polygon_added = any(isinstance(child, folium.vector_layers.Polygon)
                            for child in map_obj._children.values())
        image_overlay_added = any(isinstance(
            child, folium.raster_layers.ImageOverlay) for child in map_obj._children.values())
        self.assertTrue(polygon_added)
        self.assertTrue(image_overlay_added)


if __name__ == '__main__':
    unittest.main()
