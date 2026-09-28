// =========================================================================
// EIT Water Hackathon Munich 2026 - Challenge 4
// Google Earth Engine (GEE) Script: Real-Time Landsat Algae & LST Mapping
// Catalog: https://developers.google.com/earth-engine/datasets/catalog/landsat
//
// Target: German Freshwater Systems (e.g. Lake Constance / Bodensee)
// Datasets: 
//   - LANDSAT/LC09/C02/T1_L2 (Landsat 9 Collection 2 Tier 1 Level 2)
//   - LANDSAT/LC08/C02/T1_L2 (Landsat 8 Collection 2 Tier 1 Level 2)
// =========================================================================

// 1. Define Area of Interest
// Option A: Oder River (Odra - German-Polish Border reach near Frankfurt/Kostrzyn) [PRIMARY]
var aoi_oder = ee.Geometry.Polygon([
  [[14.40, 52.20],
   [14.75, 52.20],
   [14.75, 52.65],
   [14.40, 52.65]]
]);

// Option B: Lake Constance / Bodensee (Germany / Switzerland / Austria)
var aoi_bodensee = ee.Geometry.Polygon([
  [[9.15, 47.55],
   [9.60, 47.55],
   [9.60, 47.75],
   [9.15, 47.75]]
]);

// Set active AOI to Oder River
var aoi = aoi_oder;

Map.centerObject(aoi, 11);
Map.setOptions('SATELLITE');

// 2. Define Date Range (Summer Season)
var startDate = '2025-06-01';
var endDate = '2025-08-31';

// 3. Cloud & Water Masking function using QA_PIXEL bitmask
function maskLandsatSR(image) {
  var qa = image.select('QA_PIXEL');
  
  // Bit 3 = Dilated Cloud, Bit 4 = Cirrus, Bit 5 = Cloud, Bit 7 = Cloud Shadow
  var cloudShadowMask = qa.bitwiseAnd(1 << 4).eq(0);
  var cloudsMask = qa.bitwiseAnd(1 << 3).eq(0);
  var cirrusMask = qa.bitwiseAnd(1 << 2).eq(0);
  var mask = cloudShadowMask.and(cloudsMask).and(cirrusMask);
  
  // Scale optical bands: factor = 0.0000275, offset = -0.2
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  
  // Scale thermal Band 10: factor = 0.00341802, offset = 149.0 (Kelvin)
  // Convert to Celsius: Kelvin - 273.15
  var thermalBand = image.select('ST_B10')
    .multiply(0.00341802)
    .add(149.0)
    .subtract(273.15)
    .rename('LST_Celsius');
    
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBand, null, true)
              .updateMask(mask);
}

// 4. Compute Water & Bloom Indices Function
function addIndices(image) {
  // NDWI (Green - NIR) / (Green + NIR)
  var ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
  
  // MNDWI (Green - SWIR1) / (Green + SWIR1)
  var mndwi = image.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI');
  
  // NDVI / Algae Index (NIR - Red) / (NIR + Red)
  var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  
  // Floating Algae Index (FAI, Hu 2009)
  // FAI = NIR - [Red + (SWIR1 - Red) * ((865 - 655) / (1610 - 655))]
  var red = image.select('SR_B4');
  var nir = image.select('SR_B5');
  var swir1 = image.select('SR_B6');
  var baseline = red.add(swir1.subtract(red).multiply((865.0 - 655.0) / (1610.0 - 655.0)));
  var fai = nir.subtract(baseline).rename('FAI');
  
  // Pure water mask: MNDWI > 0.0
  var waterMask = mndwi.gt(0.0);
  
  return image.addBands([ndwi, mndwi, ndvi, fai])
              .updateMask(waterMask);
}

// 5. Ingest Landsat 8 and 9 Collections
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .map(maskLandsatSR);

var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .map(maskLandsatSR);

// Merge collections to achieve 8-day combined repeat cadence
var landsatCollection = l8.merge(l9)
  .sort('system:time_start')
  .map(addIndices);

print('Total Landsat 8/9 scenes loaded:', landsatCollection.size());

// 6. Median Summer Composite for Visualization
var medianComposite = landsatCollection.median().clip(aoi);

// Visualization Parameters
var rgbVis = {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0.0, max: 0.25, gamma: 1.3};
var lstVis = {bands: ['LST_Celsius'], min: 18.0, max: 28.0, palette: ['0000ff', '00ffff', 'ffff00', 'ff0000']};
var faiVis = {bands: ['FAI'], min: -0.02, max: 0.08, palette: ['0d47a1', '00897b', '66bb6a', 'ffee58', 'd32f2f']};
var ndviVis = {bands: ['NDVI'], min: -0.1, max: 0.4, palette: ['blue', 'white', 'yellow', 'green', 'red']};

Map.addLayer(medianComposite, rgbVis, 'Landsat True Color (RGB)');
Map.addLayer(medianComposite, lstVis, 'Lake Surface Temp (°C)', true);
Map.addLayer(medianComposite, faiVis, 'Floating Algae Index (FAI)', true);
Map.addLayer(medianComposite, ndviVis, 'Algae Biomass (NDVI)', false);

// 7. Time-Series Chart: Temperature vs Bloom Index
var tempChart = ui.Chart.image.series({
  imageCollection: landsatCollection.select(['LST_Celsius', 'NDVI', 'FAI']),
  region: aoi,
  reducer: ee.Reducer.mean(),
  scale: 30,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'Lake Constance (Bodensee): LSWT vs Bloom Indices Over Summer 2025',
  vAxes: {
    0: {title: 'Surface Temp (°C)', textStyle: {color: '#d32f2f'}},
    1: {title: 'Algae Index (NDVI / FAI)', textStyle: {color: '#2e7d32'}}
  },
  series: {
    0: {targetAxisIndex: 0, color: '#d32f2f', lineWidth: 2, pointSize: 4},
    1: {targetAxisIndex: 1, color: '#2e7d32', lineWidth: 2, pointSize: 4},
    2: {targetAxisIndex: 1, color: '#f57c00', lineWidth: 2, pointSize: 4}
  },
  hAxis: {title: 'Date', format: 'YYYY-MM-dd'}
});

print(tempChart);
