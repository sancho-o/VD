import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
import pandas as pd
from skimage import measure, morphology
from scipy import ndimage
from io import BytesIO

class VesselAnalyzer:
    def __init__(self):
        self.thresholding_methods = {
            'Otsu': lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            'Adaptive Mean': lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
            'Adaptive Gaussian': lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            'Manual': lambda img, thresh: cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
        }

    def detect_lumen_direct(self, region_image):
        filled = ndimage.binary_fill_holes(region_image).astype(np.uint8)
        holes = filled - region_image
        return np.any(holes > 0)

    def analyze_vessel_staining(self, image, threshold_method='Otsu', manual_threshold=127, min_size=100):
        if isinstance(image, (str, bytes, BytesIO)):
            # Convert uploaded file to numpy array
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            original = image

        if original is None:
            raise ValueError("Could not read the image")

        rgb_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        saturation = hsv_image[:, :, 1]

        if threshold_method == 'Manual':
            binary = self.thresholding_methods[threshold_method](saturation, manual_threshold)
        else:
            binary = self.thresholding_methods[threshold_method](saturation)

        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        vessel_mask = morphology.remove_small_objects(binary.astype(bool), min_size=min_size)
        vessel_mask = vessel_mask.astype(np.uint8) * 255

        labels = measure.label(vessel_mask)
        regions = measure.regionprops(labels)

        vessels_with_lumen = np.zeros_like(vessel_mask)
        vessels_without_lumen = np.zeros_like(vessel_mask)
        lumen_vessels = []
        non_lumen_vessels = []

        for region in regions:
            region_mask = labels == region.label
            region_image = region_mask.astype(np.uint8)

            if self.detect_lumen_direct(region_image):
                vessels_with_lumen[region_mask] = 255
                lumen_vessels.append(region)
            else:
                vessels_without_lumen[region_mask] = 255
                non_lumen_vessels.append(region)

        overlay = rgb_image.copy()
        overlay[vessels_with_lumen > 0] = [0, 255, 0]
        overlay[vessels_without_lumen > 0] = [255, 0, 0]

        measurements = {
            'Vessel Counts': {
                'vessels_with_lumen': len(lumen_vessels),
                'vessels_without_lumen': len(non_lumen_vessels),
            },
            'Areas': {
                'mean_area_with_lumen': np.mean([r.area for r in lumen_vessels]) if lumen_vessels else 0,
                'mean_area_without_lumen': np.mean([r.area for r in non_lumen_vessels]) if non_lumen_vessels else 0,
            }
        }

        return {
            'measurements': measurements,
            'processed_images': {
                'original': rgb_image,
                'binary': binary,
                'vessel_mask': vessel_mask,
                'vessels_with_lumen': vessels_with_lumen,
                'vessels_without_lumen': vessels_without_lumen,
                'overlay': overlay
            }
        }

class EnhancedVesselMetrics:
    def __init__(self):
        self.image = None
        self.vessel_mask = None
        self.regions = None

    def process_from_analyzer(self, analyzer_results):
        self.image = analyzer_results['processed_images']['original']
        self.vessel_mask = analyzer_results['processed_images']['vessel_mask']
        labels = measure.label(self.vessel_mask)
        self.regions = measure.regionprops(labels, intensity_image=cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY))

    def detect_lumen(self, region_image):
        filled = ndimage.binary_fill_holes(region_image).astype(np.uint8)
        holes = filled - region_image
        return np.any(holes > 0)

    def calculate_metrics(self):
        if self.regions is None:
            raise ValueError("No regions found. Process analyzer results first.")

        vessels_with_lumen = []
        vessels_without_lumen = []

        for region in self.regions:
            region_mask = (measure.label(self.vessel_mask) == region.label)
            region_image = region_mask.astype(np.uint8)

            if self.detect_lumen(region_image):
                vessels_with_lumen.append(region)
            else:
                vessels_without_lumen.append(region)

        total_vessels = len(self.regions)
        areas = [r.area for r in self.regions]
        image_area = self.vessel_mask.shape[0] * self.vessel_mask.shape[1]

        eccentricities = [r.eccentricity for r in self.regions]
        solidities = [r.solidity for r in self.regions]
        perimeters = [r.perimeter for r in self.regions]
        circularities = [4 * np.pi * r.area / (r.perimeter ** 2) if r.perimeter > 0 else 0
                        for r in self.regions]

        intensities = [r.mean_intensity for r in self.regions]

        centroids = [r.centroid for r in self.regions]
        if len(centroids) > 1:
            tree = cKDTree(centroids)
            distances, _ = tree.query(centroids, k=2)
            mean_nearest_neighbor = np.mean(distances[:, 1])
            vessel_density = total_vessels / image_area
        else:
            mean_nearest_neighbor = 0
            vessel_density = 0

        return {
            'basic_counts': {
                'total_vessels': total_vessels,
                'vessels_with_lumen': len(vessels_with_lumen),
                'vessels_without_lumen': len(vessels_without_lumen)
            },
            'area_measurements': {
                'total_vessel_area': np.sum(areas),
                'mean_vessel_size': np.mean(areas),
                'median_vessel_size': np.median(areas),
                'vessel_area_std': np.std(areas),
                'vessel_coverage_percentage': (np.sum(areas) / image_area) * 100
            },
            'shape_analysis': {
                'mean_eccentricity': np.mean(eccentricities),
                'mean_solidity': np.mean(solidities),
                'mean_perimeter': np.mean(perimeters),
                'mean_circularity': np.mean(circularities)
            },
            'intensity_statistics': {
                'mean_intensity': np.mean(intensities),
                'max_intensity': np.max(intensities),
                'min_intensity': np.min(intensities)
            },
            'distribution': {
                'vessel_density': vessel_density,
                'mean_nearest_neighbor': mean_nearest_neighbor
            },
            'raw_data': {
                'areas': areas,
                'eccentricities': eccentricities,
                'solidities': solidities,
                'perimeters': perimeters,
                'circularities': circularities,
                'intensities': intensities,
                'centroids': centroids
            }
        }

def create_enhanced_plots(metrics):
    fig = plt.figure(figsize=(15, 10))
    
    # Create a 2x3 grid of subplots
    gs = fig.add_gridspec(2, 3)
    
    # 1. Vessel counts
    ax1 = fig.add_subplot(gs[0, 0])
    counts = [metrics['basic_counts']['vessels_with_lumen'],
             metrics['basic_counts']['vessels_without_lumen']]
    ax1.bar(['With Lumen', 'Without Lumen'], counts)
    ax1.set_title('Vessel Counts')
    ax1.set_ylabel('Count')
    
    # 2. Area distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(metrics['raw_data']['areas'], bins=30, density=True)
    ax2.set_title('Vessel Area Distribution')
    ax2.set_xlabel('Area (pixels²)')
    
    # 3. Shape metrics
    ax3 = fig.add_subplot(gs[0, 2])
    shape_data = pd.DataFrame({
        'Eccentricity': metrics['raw_data']['eccentricities'],
        'Solidity': metrics['raw_data']['solidities'],
        'Circularity': metrics['raw_data']['circularities']
    })
    shape_data.boxplot(ax=ax3)
    ax3.set_title('Shape Metrics Distribution')
    
    # 4. Area vs Circularity scatter
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(metrics['raw_data']['areas'],
               metrics['raw_data']['circularities'],
               alpha=0.5)
    ax4.set_xlabel('Area')
    ax4.set_ylabel('Circularity')
    ax4.set_title('Area vs Circularity')
    
    # 5. Intensity distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(metrics['raw_data']['intensities'], bins=30, density=True)
    ax5.set_title('Intensity Distribution')
    ax5.set_xlabel('Mean Intensity')
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="Enhanced Vessel Analysis Tool", layout="wide")
    
    st.title("Enhanced Vessel Analysis Tool")
    st.write("Upload a vessel staining image for analysis")

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'enhanced_metrics' not in st.session_state:
        st.session_state.enhanced_metrics = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = VesselAnalyzer()

    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Parameters")
        threshold_method = st.selectbox(
            "Threshold Method",
            ['Otsu', 'Adaptive Mean', 'Adaptive Gaussian', 'Manual']
        )
        
        manual_threshold = st.slider(
            "Manual Threshold",
            0, 255, 127,
            disabled=(threshold_method != 'Manual')
        )
        
        min_size = st.slider(
            "Minimum Vessel Size",
            10, 1000, 100
        )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Original Image", use_column_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                try:
                    # Run basic analysis
                    st.session_state.results = st.session_state.analyzer.analyze_vessel_staining(
                        uploaded_file,
                        threshold_method=threshold_method,
                        manual_threshold=manual_threshold,
                        min_size=min_size
                    )
                    
                    # Run enhanced metrics
                    metrics_analyzer = EnhancedVesselMetrics()
                    metrics_analyzer.process_from_analyzer(st.session_state.results)
                    st.session_state.enhanced_metrics = metrics_analyzer.calculate_metrics()
                    
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

        if st.session_state.results is not None and st.session_state.enhanced_metrics is not None:
            results = st.session_state.results
            enhanced_metrics = st.session_state.enhanced_metrics
            
            # Create tabs for different visualizations
            tabs = st.tabs(["Basic Results", "Processed Images", "Enhanced Metrics", "Statistical Plots", "Summary Statistics"])
            
            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        results['processed_images']['overlay'], 
                        caption="Classification Overlay\nGreen: Vessels with Lumen, Red: Vessels without Lumen",
                        use_column_width=True
                    )
                
                with col2:
                    measurements = results['measurements']
                    st.subheader("Basic Analysis Results")
                    st.write("Vessel Counts:")
                    st.write(f"- Vessels with lumen: {measurements['Vessel Counts']['vessels_with_lumen']}")
                    st.write(f"- Vessels without lumen: {measurements['Vessel Counts']['vessels_without_lumen']}")
                    st.write("\nAverage Areas:")
                    st.write(f"- Vessels with lumen: {measurements['Areas']['mean_area_with_lumen']:.1f} pixels²")
                    st.write(f"- Vessels without lumen: {measurements['Areas']['mean_area_without_lumen']:.1f} pixels²")
            
            with tabs[1]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(results['processed_images']['binary'], 
                             caption="Binary Mask",
                             use_column_width=True)
                with col2:
                    st.image(results['processed_images']['vessels_with_lumen'],
                             caption="Vessels with Lumen",
                             use_column_width=True)
                with col3:
                    st.image(results['processed_images']['vessels_without_lumen'],
                             caption="Vessels without Lumen",
                             use_column_width=True)
            
            with tabs[2]:
                st.subheader("Enhanced Metrics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Area Measurements")
                    st.write(f"- Total Vessel Area: {enhanced_metrics['area_measurements']['total_vessel_area']:.2f} pixels²")
                    st.write(f"- Mean Vessel Size: {enhanced_metrics['area_measurements']['mean_vessel_size']:.2f} pixels²")
                    st.write(f"- Vessel Coverage: {enhanced_metrics['area_measurements']['vessel_coverage_percentage']:.2f}%")
                
                with col2:
                    st.write("Shape Analysis")
                    st.write(f"- Mean Circularity: {enhanced_metrics['shape_analysis']['mean_circularity']:.3f}")
                    st.write(f"- Mean Solidity: {enhanced_metrics['shape_analysis']['mean_solidity']:.3f}")
                    st.write(f"- Mean Eccentricity: {enhanced_metrics['shape_analysis']['mean_eccentricity']:.3f}")
            
            with tabs[3]:
                st.subheader("Statistical Plots")
                st.pyplot(create_enhanced_plots(enhanced_metrics))
            
            with tabs[4]:
                st.subheader("Summary Statistics")
                
                # Create an expandable section for each category
                with st.expander("Vessel Counts and Coverage", expanded=True):
                    st.write(f"- Total Vessels: {enhanced_metrics['basic_counts']['total_vessels']}")
                    st.write(f"- Vessel Coverage: {enhanced_metrics['area_measurements']['vessel_coverage_percentage']:.2f}%")
                    st.write(f"- Vessel Density: {enhanced_metrics['distribution']['vessel_density']:.4f} vessels/pixel²")
                
                with st.expander("Size Metrics", expanded=True):
                    st.write(f"- Mean Vessel Size: {enhanced_metrics['area_measurements']['mean_vessel_size']:.2f} pixels²")
                    st.write(f"- Median Vessel Size: {enhanced_metrics['area_measurements']['median_vessel_size']:.2f} pixels²")
                    st.write(f"- Size Standard Deviation: {enhanced_metrics['area_measurements']['vessel_area_std']:.2f} pixels²")
                
                with st.expander("Shape Metrics", expanded=True):
                    st.write(f"- Mean Circularity: {enhanced_metrics['shape_analysis']['mean_circularity']:.3f}")
                    st.write(f"- Mean Solidity: {enhanced_metrics['shape_analysis']['mean_solidity']:.3f}")
                    st.write(f"- Mean Eccentricity: {enhanced_metrics['shape_analysis']['mean_eccentricity']:.3f}")
                    st.write(f"- Mean Perimeter: {enhanced_metrics['shape_analysis']['mean_perimeter']:.2f} pixels")
                
                with st.expander("Distribution Metrics", expanded=True):
                    st.write(f"- Mean Nearest Neighbor Distance: {enhanced_metrics['distribution']['mean_nearest_neighbor']:.2f} pixels")
                    st.write(f"- Vessel Density: {enhanced_metrics['distribution']['vessel_density']:.4f} vessels/pixel²")
                
                with st.expander("Intensity Statistics", expanded=True):
                    st.write(f"- Mean Intensity: {enhanced_metrics['intensity_statistics']['mean_intensity']:.2f}")
                    st.write(f"- Maximum Intensity: {enhanced_metrics['intensity_statistics']['max_intensity']:.2f}")
                    st.write(f"- Minimum Intensity: {enhanced_metrics['intensity_statistics']['min_intensity']:.2f}")
                
                # Add export functionality
                if st.button("Export Results to CSV"):
                    # Create a DataFrame with all metrics
                    export_data = {
                        'Metric': [],
                        'Value': []
                    }
                    
                    # Add all metrics to the export data
                    for category, metrics in enhanced_metrics.items():
                        if category != 'raw_data':  # Skip raw data arrays
                            for metric, value in metrics.items():
                                export_data['Metric'].append(f"{category}_{metric}")
                                export_data['Value'].append(value)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(export_data)
                    
                    # Convert DataFrame to CSV
                    csv = df.to_csv(index=False)
                    
                    # Create download button
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="vessel_analysis_results.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
