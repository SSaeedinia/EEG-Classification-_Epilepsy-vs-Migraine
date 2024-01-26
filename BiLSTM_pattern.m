clc;
clear all; 
close all;
Sample_Num=36;
Sample_data=500;
Channels=11;

test_set=[1,3,5,12,13,14,15,22,25,33];
train_set=[2,4,6,7,8,9,10,11,16,17,18,19,20,21,23,24,26,27,28,29,30,31,32,34,35,36];



formatSpec = "EEG_sample%d.xlsx";
M=cell(Sample_Num,1);
kk=1;
for k=1:Sample_Num
filename=compose(formatSpec,k)
u=readmatrix(filename);

M(k)={u'};

end

XTest=M(test_set);
XTrain=M(train_set);
YCTr=[1,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3]';
YCTe=[1,1,1,2,2,2,2,3,3,3]';
YTrain=categorical(YCTr)
YTest=categorical(YCTe)
figure
plot(XTrain{1}')
xlabel("Time Step")
title("Training Observation EEG 1")
numFeatures = size(XTrain{1},1);
legend("EEG channels Data " + string(1:numFeatures),'Location','northeastoutside')
numObservations = numel(XTrain);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end
[sequenceLengths,idx] = sort(sequenceLengths);

XTrain = XTrain(idx);
YTrain = YTrain(idx);
figure
bar(sequenceLengths)
ylim([0 30])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")
%%%%%%%%%%%%%%%%%%%%%%%%%_______________________________________------------------------------------------------
miniBatchSize = 25;
inputSize = 11;
numHiddenUnits = 30;
numClasses = 3;
maxEpochs = 100;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
net = trainNetwork(XTrain,YTrain,layers,options);

numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end
[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);

YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
YTest = YTest(idx);
acc = sum(YPred == YTest)./numel(YTest)
X = XTest{1};

% figure
% plot(XTest{1}')
% xlabel("Time Step")
% xlim([0 500]);
% title("Test EEG Samples of Epilepsy ")
% numFeatures = size(XTest{1},1);
% legend("Channel Data " + string(1:numFeatures),'Location','northeastoutside')
sequenceLength = size(X,2);
idxLayer = 2;
outputSize = net.Layers(idxLayer).NumHiddenUnits;

for i = 1:sequenceLength
    features(:,i) = activations(net,X(:,i),idxLayer);
    [net, YPred(i)] = classifyAndUpdateState(net,X(:,i));
end
figure
f1=heatmap(features(1:11,1:25));
xlabel("Batch time Sequences")
ylabel("EEG Channels (Features)")
title("LSTM Activations for Epiletic Data")
saveas(f1,'Epileptic.tiff')


X = XTest{5};

figure
plot(XTest{5}')
xlabel("Time Step")
xlim([0 500]);
title("Test EEG Samples of Migraine")
numFeatures = size(XTest{1},1);
legend("Channel Data " + string(1:numFeatures),'Location','northeastoutside')
sequenceLength = size(X,2);
idxLayer = 2;
outputSize = net.Layers(idxLayer).NumHiddenUnits;

for i = 1:sequenceLength
    features(:,i) = activations(net,X(:,i),idxLayer);
    [net, YPred(i)] = classifyAndUpdateState(net,X(:,i));
end
figure
f2=heatmap(features(1:11,1:25));
xlabel("Batch time Sequences")
ylabel("EEG Channels (Features)")
title("LSTM Activations for Migraine Data")
saveas(f2,'Migraine.tiff')

X = XTest{10};

figure
plot(XTest{10}')
xlabel("Time Step")
xlim([0 500]);
title("Test EEG Samples of Normal")
numFeatures = size(XTest{1},1);
legend("Channel Data " + string(1:numFeatures),'Location','northeastoutside')
sequenceLength = size(X,2);
idxLayer = 2;
outputSize = net.Layers(idxLayer).NumHiddenUnits;

for i = 1:sequenceLength
    features(:,i) = activations(net,X(:,i),idxLayer);
    [net, YPred(i)] = classifyAndUpdateState(net,X(:,i));
end
figure
f3=heatmap(features(1:11,1:25));
xlabel("Batch time Sequences")
ylabel("EEG Channels (Features)")
title("LSTM Activations for Normal Data")
saveas(f3,'Normal.tiff')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
nx=[1,10,50,100,200,400,500];
accnx=[0.4,0.6,0.6,0.7,0.7,0.8,0.9];epochtr=50;timetr=[27,98];
figure;
plot(nx,accnx ,'g','LineWidth',3)
title('BiLSTM Network Accuracy vs units numbers (50 training epochs)')
xlabel('Units numbers')
ylabel('Accuracy')
ylim([0 1])
xlim([0 500])
%%%%%%%%%%%%%%%%%%%%%________________
nx=[10,50,100,150,200,400,500];
accnx=[0.6,0.6,0.65,0.7,0.7,0.8,0.85];epochtr=50;timetr=[27,98];
figure;
plot(nx,accnx ,'g','LineWidth',3)
title('RSM-SNN Network Accuracy vs units numbers (online training)')
xlabel('Units numbers')
ylabel('Accuracy')
ylim([0 1])
xlim([0 500])
figure;
accxx=[0.1,0.3,0.4,0.6,0.6,0.6,0.3,0.6];nxx=[10];eph=[1,10,20,30,50,80,100,200];
plot(eph,accxx ,'r','LineWidth',3)
title('BiLSTM Network Accuracy vs epoch numbers (10 hidden units)')
xlabel('epoch numbers')
ylabel('Accuracy')
ylim([0 1])
xlim([0 200])

statebl=net.Layers(2,1).HiddenState;
size(statebl)
fstate=statebl(1:numHiddenUnits ,1);
bstate=statebl(numHiddenUnits +1:2*numHiddenUnits ,1);
cells=net.Layers(2,1).CellState;
rcW=net.Layers(2,1).RecurrentWeights;
inW=net.Layers(2,1).InputWeights;
outW=net.Layers(3,1).Weights;
xv=[];yv=[];
[xw,yw]=size(rcW);
for i=1:xw
 xv=[xv,i];
end
  for j=1:yw
        
      yv=[yv,j] ; 
  end
    
  figure;
 mesh(rcW)
  title('Hidden state recurrent weights')
xlabel('Hidden connections')
ylabel('Hidden BiLSTM units')

[xw,yw]=size(inW);
for i=1:xw
 xv=[xv,i];
end
  for j=1:yw
        
      yv=[yv,j] ; 
  end
    
  figure;
 mesh(inW')
  title('Input  weights')
xlabel('Input to Hiddent layers connections')
ylabel('Input Freatures (EEG channel Data)')

figure;
wih=inW'*rcW;
s=mesh(wih)
s.FaceColor = 'flat';
  title('Input to Hidden Units Weights')
xlabel('Input EEG channels Features')
ylabel('Hidden Units')
figure;
subplot(2,1,1)
s=mesh(outW(:,1:numHiddenUnits))
s.FaceColor = 'flat';
  title('Forward Hidden to output  Weights')
xlabel('Hidden units (forward and Backward)')
ylabel('Outout Neurons')
subplot(2,1,2)
s=mesh(outW(:,numHiddenUnits +1:2*numHiddenUnits))
s.FaceColor = 'flat';
  title('Backward Hidden to output  Weights')
xlabel('Hidden units (forward and Backward)')
ylabel('Outout Neurons')
nrow=norm(outW);
c1=sum(outW(1,:));
c2=sum(outW(2,:));
c3=sum(outW(3,:));
ac1=c1/nrow 
ac2=c2/nrow 
ac3=c3/nrow 

