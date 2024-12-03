#!/bin/bash
#PBS -N GaDIFF_02
#PBS -A P03010039
#PBS -l walltime=03:00:00
#PBS -o Gather_01_TrainDiffusion.out
#PBS -e Gather_01_TrainDiffusion.out
#PBS -q casper
#PBS -l select=1:ncpus=64:mem=470GB:ngpus=4 -l gpu_type=a100
#PBS -m a
#PBS -M wchapman@ucar.edu

# qsub -I -q main -A P54048000 -l walltime=01:00:00 -l select=1:ncpus=1:mem=12GB:ngpus=4 -l gpu_type=a100
# qsub -I -q casper -A P03010039 -l walltime=01:00:00 -l select=1:ncpus=1:mem=30GB:ngpus=1 -l gpu_type=a100

#accelerate config
module load conda
conda activate LuRain

# Number of samples to pick
SAMPLES=80
MONTH=2

# Path to the CSV file
csv_file="./co2vmr_month.csv"

# Extract and sort co2vmr values corresponding to the chosen month
co2_values=$(awk -F, -v month="$MONTH" '$2 == month {print $1}' "$csv_file" | sort -n)

# Set the threshold X value for CO2
CO2_THRESHOLD=0.0005697441818383968

# Convert the sorted co2_values to an array
IFS=$'\n' read -rd '' -a co2_array <<<"$co2_values"

# Number of co2vmr values available for the given month
total_co2=${#co2_array[@]}

# Print the total number of co2 values for the specified month
echo "Total CO2 values for month $MONTH: $total_co2"

# Check if there are enough values to sample
if [ "$total_co2" -lt "$SAMPLES" ]; then
  echo "Not enough CO2 values for month $MONTH. Available: $total_co2, Requested: $SAMPLES"
  exit 1
fi

# Calculate the step size for even distribution
step=$(echo "$total_co2 / $SAMPLES" | bc)

# Loop through and select evenly spaced co2vmr values
for ((i=0; i<SAMPLES; i++)); do
  index=$(echo "$i * $step" | bc)
  co2=${co2_array[$index]}
  # Check if the CO2 value is below the threshold
  if (( $(echo "$co2 > $CO2_THRESHOLD" | bc -l) )); then
    printf "Skipping CO2 value: %s as it is below the threshold of %d\n" "$co2" "$CO2_THRESHOLD"
    continue
  fi
  echo accelerate launch Gen_Data.py --month "$MONTH" --co2 "$co2"
  #python Gen_Data.py --month "$MONTH" --co2 "$co2"
done

accelerate launch Gen_Data.py --month 2 --co2 0.0004049043903089039 --run_num 3934
accelerate launch Gen_Data.py --month 2 --co2 0.00061969236097592 --run_num 3934
accelerate launch Gen_Data.py --month 2 --co2 0.0008001525352122135 --run_num 3934
