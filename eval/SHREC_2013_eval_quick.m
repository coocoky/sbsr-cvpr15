function [Precision, Recall, Av_Precision, P_points, R_points, C, Pmat] = SHREC_2013_eval_quick(RANK_DIR, testname, queries, results, disp_flag)

if (nargin < 4)
  disp_flag = 0;
end

number_of_queries=length(queries);
number_of_target=size(results,2);

match = bsxfun(@eq, queries, results);
C = sum(match, 2);
  
P_points=zeros(number_of_queries,max(C));
R_points=zeros(number_of_queries,max(C));

Av_Precision=zeros(1,number_of_queries); 
NN=zeros(1,number_of_queries);
FT=zeros(1,number_of_queries);
ST=zeros(1,number_of_queries);
dcg=zeros(1,number_of_queries);
E=zeros(1,number_of_queries);

filename=fullfile(RANK_DIR, sprintf('Stats_%s.txt', testname));
fid=fopen(filename,'w');
fprintf(fid,'        NN     FT     ST      E       DCG\n');

for qqq=1:number_of_queries    

    R = results(qqq, :);
    q = queries(qqq);
    
    G = (R == q);
    G_sum=cumsum(G); 
    
    for rec=1:C(qqq)
        R_points(qqq, rec)=find((G_sum==rec),1);
    end;
    
    P_points(qqq,1:C(qqq))=G_sum(R_points(qqq,1:C(qqq)))./R_points(qqq,1:C(qqq));
    Av_Precision(qqq)=mean(P_points(qqq,:));
    
    NN(qqq)=G(1);
    FT(qqq)=G_sum(C(qqq))/C(qqq);
    ST(qqq)=G_sum(2*C(qqq))/C(qqq);
    P_32=G_sum(32)/32;
    R_32=G_sum(32)/C(qqq);
    
    if (P_32==0)&&(R_32==0);
        E(qqq)=0;
    else
        E(qqq)=2*P_32*R_32/(P_32+R_32);
    end;
    
    NORM_VALUE=1+sum(1./log2([2:C(qqq)]));
    dcg_i=(1./log2([2:length(R)])).*G(2:end);
    dcg_i=[G(1);dcg_i(:)];
    dcg(qqq)=sum(dcg_i)/NORM_VALUE;
    
    fprintf(fid,'No.%d: %2.3f\t %2.3f\t %2.3f\t %2.3f\t %2.3f\n',qqq, NN(qqq),FT(qqq),ST(qqq),E(qqq),dcg(qqq));
end;
fclose(fid);    

NN_av=mean(NN);
FT_av=mean(FT);
ST_av=mean(ST);
dcg_av=mean(dcg);
E_av=mean(E);
Mean_Av_Precision=mean(Av_Precision) ;


filename=fullfile(RANK_DIR, sprintf('PR_%s.txt', testname));
fid=fopen(filename,'w');
[Precision, Pmat]=calcAvgPerf(P_points, C, number_of_queries, filename);
fclose(fid);
Recall=[1:20]/20;

Mean_Av_Precision = mean(Precision);

if disp_flag
  fprintf('NN: %f \tFT: %f \tST: %f \tE_av: %f \tdcg: %f \tmAP: %f \n', NN_av, FT_av, ST_av, E_av, dcg_av, Mean_Av_Precision);
end
