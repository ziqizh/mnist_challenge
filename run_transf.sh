echo A1 attack A1
python measure-transferability.py --target-model data-model/model_r10000/checkpoint-99600 --path data-adv/model.r10000.checkpoint-99600.A1.adv.npy

echo A2 attack A2
python measure-transferability.py --target-model data-model/model_r10000/checkpoint-99300 --path data-adv/model.r10000.checkpoint-99300.A2.adv.npy

echo A3 attack A3
python measure-transferability.py --target-model data-model/model_r10000/checkpoint-99000 --path data-adv/model.r10000.checkpoint-99000.A3.adv.npy

echo B attack B
python measure-transferability.py --target-model data-model/model_r20000/checkpoint-99900 --path data-adv/model.r20000.checkpoint-99900.B.adv.npy

echo C attack C
python measure-transferability.py --target-model data-model/model_r30000/checkpoint-99900 --path data-adv/model.r30000.checkpoint-99900.C.adv.npy

echo D attack D
python measure-transferability.py --target-model data-model/model_r40000/checkpoint-99900 --path data-adv/model.r40000.checkpoint-99900.D.adv.npy


echo A1 attack A
python measure-transferability.py --target-model data-model/model_r10000/checkpoint-99900 --path data-adv/model.r10000.checkpoint-99600.A1.adv.npy

echo A2 attack A
python measure-transferability.py --target-model data-model/model_r10000/checkpoint-99900 --path data-adv/model.r10000.checkpoint-99300.A2.adv.npy

echo A3 attack A
python measure-transferability.py --target-model data-model/model_r10000/checkpoint-99900 --path data-adv/model.r10000.checkpoint-99000.A3.adv.npy

echo B attack A
python measure-transferability.py --target-model data-model/model_r10000/checkpoint-99900 --path data-adv/model.r20000.checkpoint-99900.B.adv.npy

echo C attack A
python measure-transferability.py --target-model data-model/model_r10000/checkpoint-99900 --path data-adv/model.r30000.checkpoint-99900.C.adv.npy

echo D attack A
python measure-transferability.py --target-model data-model/model_r10000/checkpoint-99900 --path data-adv/model.r40000.checkpoint-99900.D.adv.npy


