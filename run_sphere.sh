for l in 0 0.9
do
	for k in "sgd" "adam"
	do
		sbatch start.sh "clean" 0.1 10 $k
		sbatch start.sh "truemax" 0.1 10 $k

		for i in 1 5 10 50 100 
		do
			for j in 0.1 0.01
			do
				sbatch start.sh "adv" $j $i $k
			done
		done
	done
done
