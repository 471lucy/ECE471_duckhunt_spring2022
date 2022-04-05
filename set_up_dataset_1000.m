function [D_1000_train, D_1000_test] = set_up_dataset_1000()
D_1000 = load('dataset_1000') %D_1000.arr is 1000x8100 dataset

%first 500 are ducks, last 500 are not ducks
%1 for positive duck class, -1 for negative duck class binary
%classification


D_1000_ducks_train = D_1000.arr(1:400,:);
D_1000_ducks_test = D_1000.arr(401:500,:);
size(D_1000_ducks_train); %400 x 8101
size(D_1000_ducks_test); % 100 x 8101

D_1000_noducks_train = D_1000.arr(501:900,:);
D_1000_noducks_test = D_1000.arr(901:1000,:);
size(D_1000_noducks_train); %400 x 8101
size(D_1000_noducks_test); % 100 x 8101

%set up so the outputs are the last row training set
D_1000_train = [D_1000_ducks_train;D_1000_noducks_train];
D_1000_train = D_1000_train';
size(D_1000_train);
y_pos = ones(1, 400);
y_neg = -1.*ones(1, 400);
y_train = [y_pos, y_neg];
size(y_train);
D_1000_train = [D_1000_train; y_train];
size(D_1000_train); % 8101 x 800



%set up test set
D_1000_test = [D_1000_ducks_test;D_1000_noducks_test];
D_1000_test = D_1000_test';
size(D_1000_test);
y_pos_test = ones(1, 100);
y_neg_test = -1.*ones(1, 100);
y_test = [y_pos_test, y_neg_test];
size(y_test);
D_1000_test = [D_1000_test; y_test];
size(D_1000_test); % 8101 x 800



end