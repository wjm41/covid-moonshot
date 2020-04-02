for i in *_updated/;
do
babel -ixyz ${i}*.xyz -ocan ${i}smiles_updated.can
cat ${i}smiles_updated.can >> smiles_updated.can
echo \ >> smiles_updated.can
done

