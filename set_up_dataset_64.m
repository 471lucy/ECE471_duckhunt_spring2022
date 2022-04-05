function [D_64_train, D_64_test] = set_up_dataset_64()
D_64 = load('dataset_64') %D_64.arr is 1450x1764 dataset - 350 are images of ducks


D_64_ducks_train = D_64.arr(1:400,:);
D_64_ducks_test = D_64.arr(401:500,:);
size(D_64_ducks_train); 
size(D_64_ducks_test);

D_64_noducks_train = D_64.arr(501:1400,:);
D_64_noducks_test = D_64.arr(1401:1600,:);
size(D_64_noducks_train); 
size(D_64_noducks_test); 

%set up so the outputs are the last row training set
D_64_train = [D_64_ducks_train;D_64_noducks_train];
D_64_train = D_64_train';
size(D_64_train);
y_pos = ones(1, 400);
y_neg = -1.*ones(1, 900);
y_train = [y_pos, y_neg];
size(y_train);
D_64_train = [D_64_train; y_train];
size(D_64_train); 


%set up test set
D_64_test = [D_64_ducks_test;D_64_noducks_test];
D_64_test = D_64_test';
size(D_64_test);
y_pos_test = ones(1, 100);
y_neg_test = -1.*ones(1, 200);
y_test = [y_pos_test, y_neg_test];
size(y_test);
D_64_test = [D_64_test; y_test];
size(D_64_test);



end