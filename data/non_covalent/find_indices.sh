for i in *.xyz;
do
echo ${i} | cut -c7-10 >> indices.txt
done
