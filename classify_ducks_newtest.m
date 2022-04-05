function [] = classify_ducks_newtest(ws)

D_newtest = load('dataset_new_test')

%first 19 are ducks

%training data set classification

D_newtest = D_newtest.arr';
D_test = [D_newtest(1:8100,:);ones(1,730)];

%test dataset classification
total_pos_test = 19;
total_neg_test = 711;
pos_test = 0;
neg_test = 0;

%positive duck train
for i = 1:1:19
    class = sign(ws'*D_test(:,i));
    if class > 0
        pos_test = pos_test + 1;
    end
end
percent_correct_pos_test = (pos_test/total_pos_test) * 100

%negative duck train
for i = 20:1:730
    class = sign(ws'*D_test(:,i));
    if class < 0
        neg_test = neg_test + 1;
    end
end
percent_correct_neg_test = (neg_test/total_neg_test) * 100









end