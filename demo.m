% DEMO for testing MEDA on Office+Caltech10 datasets
clear all
clc
addpath('.\liblinear\');
%addpath('.\utility\');

result={'target', 'F_measure','g_measure','AUC'};

%company
    
    [project,L]=loadproject();
    for t1=1:L
       for itea=1:5
        for s1=1:L
             if s1==t1
                    continue;
             end
                    ratio=0.9;
                    src=project(s1).Dataset;
      
                sname=project(s1).Name;
                src=src(randperm(size(src,1)),:);
                tgt=project(t1).Dataset;
                tname=project(t1).Name;
                tgt=tgt(randperm(size(tgt,1)),:);
                
                fs=src(:,1:size(src,2)-1);
                fs = fs ./ repmat(sum(fs,2),1,size(fs,2));
                Xs = zscore(fs,1);
                Ys = src(:,size(src,2));
                Ys=Ys+1;
                legth=size(fs,1);leg_train=ceil(ratio*legth);
                Xs=Xs(1:leg_train,:);Ys=Ys(1:leg_train,:);
                
                
                ft=tgt(:,1:size(tgt,2)-1);
                ft = ft ./ repmat(sum(ft,2),1,size(ft,2));
                Xt = zscore(ft,1);
                Yt = tgt(:,size(tgt,2));
                Yt=Yt+1;
              
                options.d=5;%manifold
                options.dim =3;%A
                options.rho = 0.1;
                options.lambda = 0.1;       
                options.p=5;
                options.mu=0.6;
                
                [~,pre,result] = MEDA(sname,tname,Xs,Ys,Xt,Yt,options,result);
                
               
                
        end
       end
        end
  
function [A_project,i] = loadproject()
A_CompanyName='data';
 A_DatasetNames = {'ant','camel', 'ivy','lucene','poi','synapse','velocity','xalan','xerces'} ;
for i = 1 : size(A_DatasetNames, 2)
load([A_CompanyName, '/', A_DatasetNames{i}, '.mat']);
 A_project(i).Dataset = eval(A_DatasetNames{i});
 A_project(i).Name = A_DatasetNames{i};
end
end
