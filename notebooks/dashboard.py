import os
import shutil
import json
import zipfile
from math import ceil, floor
import concurrent.futures
from tqdm.notebook import tqdm
import requests
from ipyleaflet import Map, basemaps
from osgeo import gdal
import numpy as np
import zarr
import xarray as xr
from ipygany import Scene, PolyMesh, IsoColor, Component
from ipyleaflet import DrawControl
import matplotlib.pyplot as plt
from IPython.display import display, FileLink


class Dashboard:

    def __init__(self, m, notif, dem2d, dem3d):
        self.map = m
        self.notif = notif
        self.dem2d = dem2d
        self.dem3d = dem3d
        self.tile_dir = 'dem_tiles'

    def start(self):
        self.draw_control = DrawControl()
        self.draw_control.polygon = {}
        self.draw_control.polyline = {}
        self.draw_control.circlemarker = {}
        self.draw_control.rectangle = {
            'shapeOptions': {
                'fillOpacity': 0.5
            }
        }
        self.draw_control.on_draw(self.show_dem)
        self.map.add_control(self.draw_control)

    def show_dem(self, *args, **kwargs):
        self.dem2d.clear_output()
        self.dem3d.clear_output()
        lonlat = self.draw_control.last_draw['geometry']['coordinates'][0]
        lats = [ll[1] for ll in lonlat]
        lons = [ll[0] for ll in lonlat]
        lt0, lt1 = min(lats), max(lats)
        ln0, ln1 = min(lons), max(lons)
        os.makedirs(self.tile_dir, exist_ok=True)
        with open(self.tile_dir + '.json') as f:
            tiles = json.loads(f.read())
        lat = lat0 = floor(lt0 / 5) * 5
        lon = lon0 = floor(ln0 / 5) * 5
        lat1 = ceil(lt1 / 5) * 5
        lon1 = ceil(ln1 / 5) * 5
        ny = int(round((lat1 - lat0) / (5 / 6000)))
        nx = int(round((lon1 - lon0) / (5 / 6000)))
        zarr_path = os.path.join(self.tile_dir + '.zarr')
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        z = zarr.open(zarr_path, mode='w', shape=(ny, nx), chunks=(6000, 6000), dtype='float32')
        done = False
        tasks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while not done:
                tasks.append(executor.submit(self.create_zarr_chunk, lat, lon, tiles, lat0, lat1, lon0, lon1, z))
                lon += 5
                if lon >= ln1:
                    lon = lon0
                    lat += 5
                    if lat >= lt1:
                        done = True
            for task in tasks:
                task.result()
        y = np.arange(lat0, lat1, 5 / 6000)
        x = np.arange(lon0, lon1, 5 / 6000)
        dem = xr.DataArray(z, coords=[y, x], dims=['y', 'x']).sel(y=slice(lt0, lt1), x=slice(ln0, ln1))
        self.show_dem2d(dem)
        self.show_dem3d(dem)

    def show_dem2d(self, dem):
        nr, nc = len(dem.y), len(dem.x)
        lon, lat = np.meshgrid(dem.x * np.pi / 180, dem.y * np.pi / 180, sparse=True)
        self.vertices = np.empty((nr, nc, 3), dtype='float32')
        alt = np.where(np.isnan(dem.values), 0, dem.values)
        r = 6371e3  # Earth's radius in meters
        f = alt + r
        self.vertices[:, :, 0] = np.sin(np.pi / 2 - lat) * np.cos(lon) * f
        self.vertices[:, :, 1] = np.sin(np.pi / 2 - lat) * np.sin(lon) * f
        self.vertices[:, :, 2] = np.cos(np.pi / 2 - lat) * f
        self.vertices = self.vertices.reshape(nr * nc, 3)

        self.notif.clear_output()
        np.savez('dem.npz', dem=self.vertices)
        local_file = FileLink('dem.npz', result_html_prefix="Click here to download: ")
        with self.notif:
            display(local_file)

        fig = dem.plot.imshow(vmin=np.nanmin(dem), vmax=np.nanmax(dem))
        with self.dem2d:
            display(fig)
            plt.show()

    def show_dem3d(self, dem):
        nr, nc = len(dem.y), len(dem.x)
        triangle_indices = np.empty((nr - 1, nc - 1, 2, 3), dtype='uint32')
        r = np.arange(nr * nc, dtype='uint32').reshape(nr, nc)
        triangle_indices[:, :, 0, 0] = r[:-1, :-1]
        triangle_indices[:, :, 1, 0] = r[:-1, 1:]
        triangle_indices[:, :, 0, 1] = r[:-1, 1:]
        triangle_indices[:, :, 1, 1] = r[1:, 1:]
        triangle_indices[:, :, :, 2] = r[1:, :-1, None]
        triangle_indices.shape = (-1, 3)

        height_component = Component(name='value', array=dem.values)

        mesh = PolyMesh(
            vertices=self.vertices,
            triangle_indices=triangle_indices,
            data={'height': [height_component]}
        )

        colored_mesh = IsoColor(mesh, input=('height', 'value'), min=np.nanmin(dem.values), max=np.nanmax(dem.values))

        with self.dem3d:
            display(Scene([colored_mesh]))

    def create_zarr_chunk(self, lat, lon, tiles, lat0, lat1, lon0, lon1, z):
        if lat < 0:
            fname = 's'
        else:
            fname = 'n'
        fname += str(abs(lat)).zfill(2)
        if lon < 0:
            fname += 'w'
        else:
            fname += 'e'
        fname += str(abs(lon)).zfill(3)
        fname += '_con_grid.zip'
        url = ''
        for continent in tiles:
            if fname in tiles[continent][1]:
                url = tiles[continent][0] + fname
                break
        if url:
            filename = os.path.basename(url)
            name = filename[:filename.find('_grid')]
            adffile = os.path.join(self.tile_dir, name, name, 'w001001.adf')
            zipfname = os.path.join(self.tile_dir, filename)

            if not os.path.exists(adffile):
                r = requests.get(url, stream=True)
                with open(zipfname, 'wb') as f:
                    total_length = int(r.headers.get('content-length'))
                    self.notif.clear_output()
                    with self.notif:
                        for chunk in tqdm(r.iter_content(chunk_size=1024), total=ceil(total_length/1024)):
                            if chunk:
                                f.write(chunk)
                                f.flush()
                zip = zipfile.ZipFile(zipfname)
                zip.extractall(self.tile_dir)

            dem = gdal.Open(adffile)
            geo = dem.GetGeoTransform()
            ySize, xSize = dem.RasterYSize, dem.RasterXSize
            dem = dem.ReadAsArray()
            # data is padded into a 6000x6000 array (some tiles may be smaller)
            array_5x5 = np.full((6000, 6000), -32768, dtype='int16')
            y0 = int(round((geo[3] - lat - 5) / geo[5]))
            y1 = 6000 - int(round((lat - (geo[3] + geo[5] * ySize)) / geo[5]))
            x0 = int(round((geo[0] - lon) / geo[1]))
            x1 = 6000 - int(round(((lon + 5) - (geo[0] + geo[1] * xSize)) / geo[1]))
            array_5x5[y0:y1, x0:x1] = dem
            array_5x5 = np.where(array_5x5==-32768, np.nan, array_5x5.astype('float32'))
            y0 = z.shape[0] + (lat - lat1) // 5 * 6000
            y1 = y0 + 6000
            x0 = (lon - lon0) // 5 * 6000
            x1 = x0 + 6000
            z[y0:y1, x0:x1] = array_5x5[::-1]
