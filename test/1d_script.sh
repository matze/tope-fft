set -e

#echo "1D Tests"
#----------- Radix 2
for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do
	bin/1d $(( 2 ** $j )) 2> /dev/null
done

#----------- Radix 4
#for j in 12 
#do
#	bin/1d $(( 4 ** $j )) 2> /dev/null
#done

#----------- Radix 8
#for j in 8
#do
#	bin/1d $(( 8 ** $j )) 2> /dev/null
#done

#holdMax=1000
#for (( j = 10 ; j <= 1000; j+=10 )) 
#do
#	./a.out 8388608 1 1 300 uniform uniform $j no 2> /dev/null     
#done
