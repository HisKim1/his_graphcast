# Spectral Analysis of Atmospheric Energy - Based on Augier & Lindborg (2013)
#!/bin/bash

# Configuration
WORK_DIR="./work"
OUTPUT_DIR="./output"
PRES_LEVEL=300  # 300 hPa in Pa
TEMP_PREFIX="temp"

mkdir -p $WORK_DIR $OUTPUT_DIR

# Function to process a single ensemble member or timestep
process_wind_components() {
    local input_file=$1
    local member_id=$2
    local output_prefix=$3

    # 1. Extract wind components at specified pressure level
    # Note: Using -sellevel instead of -selhlevel for pressure level selection
    cdo -remapcon,r360x181 -sellevel,$PRES_LEVEL -selname,u_component_of_wind ${input_file} ${WORK_DIR}/${TEMP_PREFIX}_u_${member_id}.nc
    cdo -remapcon,r360x181 -sellevel,$PRES_LEVEL -selname,v_component_of_wind ${input_file} ${WORK_DIR}/${TEMP_PREFIX}_v_${member_id}.nc

    # 2. Compute vorticity and divergence
    # This follows Augier & Lindborg's separation of rotational and divergent components
    cdo -sp2gp ${WORK_DIR}/${TEMP_PREFIX}_u_${member_id}.nc ${WORK_DIR}/${TEMP_PREFIX}_vor_${member_id}.nc
    cdo -sp2gp ${WORK_DIR}/${TEMP_PREFIX}_v_${member_id}.nc ${WORK_DIR}/${TEMP_PREFIX}_div_${member_id}.nc
}

# Function to compute KE spectrum
compute_ke_spectrum() {
    local input_prefix=$1
    local output_file=$2
    
    # 1. Calculate KE = (u² + v²)/2
    cdo -O -mul ${WORK_DIR}/${input_prefix}_u.nc ${WORK_DIR}/${input_prefix}_u.nc ${WORK_DIR}/${TEMP_PREFIX}_u2.nc
    cdo -O -mul ${WORK_DIR}/${input_prefix}_v.nc ${WORK_DIR}/${input_prefix}_v.nc ${WORK_DIR}/${TEMP_PREFIX}_v2.nc
    cdo -O -add ${WORK_DIR}/${TEMP_PREFIX}_u2.nc ${WORK_DIR}/${TEMP_PREFIX}_v2.nc ${WORK_DIR}/${TEMP_PREFIX}_ke.nc
    cdo -O -mulc,0.5 ${WORK_DIR}/${TEMP_PREFIX}_ke.nc ${WORK_DIR}/${TEMP_PREFIX}_ke_final.nc

    # 2. Compute spherical harmonic decomposition
    # This follows equation (13) in Augier & Lindborg
    cdo -sp2gp -sinfo ${WORK_DIR}/${TEMP_PREFIX}_ke_final.nc ${output_file}
}

# Function to compute DKE spectrum from ensemble
compute_dke_spectrum() {
    local input_pattern=$1
    local output_file=$2

    # 1. Compute ensemble mean
    cdo ensmean ${input_pattern} ${WORK_DIR}/${TEMP_PREFIX}_mean.nc

    # 2. Compute deviations from mean
    for file in ${input_pattern}; do
        member_id=$(basename $file .nc)
        cdo sub $file ${WORK_DIR}/${TEMP_PREFIX}_mean.nc ${WORK_DIR}/${TEMP_PREFIX}_dev_${member_id}.nc
    done

    # 3. Compute variance
    cdo ensvar ${input_pattern} ${WORK_DIR}/${TEMP_PREFIX}_var.nc

    # 4. Compute spherical harmonic decomposition of variance
    cdo -sp2gp -sinfo ${WORK_DIR}/${TEMP_PREFIX}_var.nc ${output_file}
}

# Main processing function
main() {
    local gc_file=$1  # GraphCast output
    local ens_pattern=$2  # ECMWF ensemble pattern

    # Process GraphCast output
    echo "Processing GraphCast output..."
    process_wind_components $gc_file "gc" "graphcast"
    compute_ke_spectrum "graphcast" "${OUTPUT_DIR}/gc_ke_spectrum.txt"

    # Process ensemble data
    # echo "Processing ensemble data..."
    # for ens_file in $ens_pattern; do
    #     member_id=$(basename $ens_file .nc)
    #     process_wind_components $ens_file $member_id "ensemble"
    # done
    # compute_dke_spectrum "${WORK_DIR}/${TEMP_PREFIX}_ensemble_*.nc" "${OUTPUT_DIR}/ensemble_dke_spectrum.txt"

    # Clean up
    rm -f ${WORK_DIR}/${TEMP_PREFIX}*
}

# Execute with error handling
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 graphcast_file.nc 'ensemble_pattern.nc'"
    exit 1
fi

main "$@"