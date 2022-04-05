function [] = classify_ducks_64(ws,D_train,D_test)

%training data set classification

%need percent correctly identified

D_train = [D_train(1:1764,:);ones(1,1300)];
size(D_train(:,1))
size(ws)

D_test = [D_test(1:1764,:);ones(1,300)];
total_pos_train = 400;
total_neg_train = 900;
pos_train = 0;
neg_train = 0;

%positive duck train
for i = 1:1:400
    class = sign(ws'*D_train(:,i));
    if class > 0
        pos_train = pos_train + 1;
    end
end
percent_correct_pos_train = (pos_train/total_pos_train) * 100

%negative duck train
for i = 401:1:1300
    class = sign(ws'*D_train(:,i));
    if class < 0
        neg_train = neg_train + 1;
    end
end

percent_correct_neg_train = (neg_train/total_neg_train) * 100

%test dataset classification
total_pos_test = 100;
total_neg_test = 200;
pos_test = 0;
neg_test = 0;

%positive duck train
for i = 1:1:100
    class = sign(ws'*D_test(:,i));
    if class > 0
        pos_test = pos_test + 1;
    end
end
percent_correct_pos_test = (pos_test/total_pos_test) * 100

%negative duck train
for i = 101:1:300
    class = sign(ws'*D_test(:,i));
    if class < 0
        neg_test = neg_test + 1;
    end
end
percent_correct_neg_test = (neg_test/total_neg_test) * 100

end