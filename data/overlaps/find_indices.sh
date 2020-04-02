for i in */;
do
babel -ixyz ${i}*.xyz -ocan ${i}smiles.can
cat ${i}smiles.can >> smiles.can
echo \ >> smiles.can
done

