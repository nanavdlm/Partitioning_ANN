clear
close all

% TO_BE_SET. Path where the codes are present. In the same folder there must
% be a subfolder named "Input" with the original input data and the subfolder
% prepared by the first code run.
main_path = 'C:\Users\rdaelman\OneDrive - UGent\Documents\GitHub\Partitioning_ANN\';
addpath(main_path);

path_input_data = [main_path 'Prepared_data\'];
lis_of_ff = dir([path_input_data '*_data4cann_2v0.mat*']);
path_input_selected = [main_path 'Test_output\MaxDriver\Selected\'];
lis_of_sel = dir([path_input_selected '*_selected_anns.mat*']);
Sites_fin = {};
Sites2elab = {};
in_s = [];
for n_s = 1:numel(lis_of_ff)
    clear tSiteChar
    tSiteChar = [char(lis_of_ff(n_s,1).name(1:2)) '_' char(lis_of_ff(n_s,1).name(4:6))];
    Sites2elab(n_s,1)={tSiteChar};
    in_s = [in_s; n_s];
    clear tSiteChar
end
clear n_s
% For each site to elaborate
for n_s = 1:numel(Sites2elab)
    
    clear C_flux_Tab C_flux_Tab_Qc C_flux_head Drivers_Tab Drivers_Tab_Qc Drivers_head Drivers_needed_head ModelsStruct OutGPP_NNcust
    clear OutGPP_NNcust_DT2t OutNEE_tset_NNcust OutNEE_tset_NNcust_DT2t OutNEE_tset_NNcust_NT OutReco_NNcust OutReco_NNcust_DT2t
    clear OutReco_NNcust_NT Rad_data Tab_mod TimeDate TimeDate_Qc TimeDate_head Variables_Tab Variables_Tab_Qc
    clear Variables_head X X2 X2_head X2qc X_head ans data data_header fnames i2rem iNeeDaytime iNeeNightime
    clear iPr ia ib ifin l main_driver_qc
    clear ndset prov_d_head_photo prov_d_head_reco
    clear prov_d_head_switchoff repetition start_rep s_name tRad_data tRad_head tX2 tX2_head tX2qc tX_format tYtr t_head tab_n_c tn tstep uy year_done  
    
    clear s_name
    s_name = char(Sites2elab(n_s));
    
    % Load the data
    load([path_input_data s_name '_data4cann_2v0.mat'])
    load([path_input_selected s_name '_selected_anns.mat'])
    for m3 = 1:2
        for i=1:5
            writematrix(eval(['SelectedModelsStruct.y', num2str(uy(m3)),'.ALL.rep(i).CANN.IW{1,1}']), [num2str(uy(m3)) '_' num2str(i) '_IW1'])
            writematrix(eval(['SelectedModelsStruct.y', num2str(uy(m3)),'.ALL.rep(i).CANN.IW{3,2}']), [num2str(uy(m3)) '_' num2str(i) '_IW2'])
            writematrix(eval(['SelectedModelsStruct.y', num2str(uy(m3)),'.ALL.rep(i).CANN.IW{5,3}']), [num2str(uy(m3)) '_' num2str(i) '_IW3'])
            writematrix(eval(['SelectedModelsStruct.y', num2str(uy(m3)),'.ALL.rep(i).CANNPara.perm']), [num2str(uy(m3)) '_' num2str(i) '_Perm'])

        end
    end

    for m3 = 1:2
        for i=1:5
            r=eval(['SelectedModelsStruct.y', num2str(uy(m3)),'.ALL.rep(i).CANNPara.r']);
            rmse=eval(['SelectedModelsStruct.y', num2str(uy(m3)),'.ALL.rep(i).CANNPara.rmse']);
            mae=eval(['SelectedModelsStruct.y', num2str(uy(m3)),'.ALL.rep(i).CANNPara.mae']);
            m=eval(['SelectedModelsStruct.y', num2str(uy(m3)),'.ALL.rep(i).CANNPara.m']);
            perf=[r rmse mae m]
            writematrix(perf, [num2str(uy(m3)) '_' num2str(i) '_Perf']);
            clear r rmse mae m perf
        end
    end
    i_nan = isnan(Ytr);
    RECO_ANN_or = RECO_ANN;
    RECO_ANN_or(i_nan,:)=NaN;
    GPP_ANN_or = GPP_ANN;
    GPP_ANN_or(i_nan,:)=NaN;  
    OutNEE=[RECO_ann_MaxD_or RECO_ANN_or] - [GPP_ann_MaxD_or GPP_ANN_or];
    for n = 1:numel(uy)
        % Find the data of the year uy(n)
        clear i1
        i1 = TimeDate(:,1)==uy(n);
        % Pick-up the NEE estimated by NNC-part (OutNEE_NNcust_ALL) and the
        % one measured by EC (Ytr)
        clear t_nee_all t_nee_ref
        t_nee_all = OutNEE(i1,:);
        t_nee_ref = repmat(Ytr(i1,:),1,size(t_nee_all,2));
        % Find where there finite value of NEEs to calculate the model's
        % efficiency
        clear i2
        i2 = isfinite(prod([t_nee_all, t_nee_ref],2));
        % ME = 1-mef_num./mef_den; mef_num = mean((Y_mod-Y_ref)^2); mef_den = mean((mean(Y_ref)-Y_ref)^2)
        mef_num = mean(((t_nee_all(i2,:)-t_nee_ref(i2,:)).^2),1);
        mef_den = mean(((repmat(nanmean(t_nee_ref(i2,:),1),size(t_nee_ref(i2,:),1),1)-t_nee_ref(i2,:)).^2),1);
        mef_vec = ones(size(mef_num))-(mef_num./mef_den);


        writematrix(mef_vec, [num2str(uy(n)) '_Eff2']);
        clear i2
        clear mef_num mef_den mef_vec
        clear t_nee_ref t_nee_all 
    end
    clear i_nan

    













    % Pick-up radiation data    
    Rad_data_head = {'SW_IN_POT';'SW_IN_f'};
    Rad_data = NaN(size(X2,1),numel(Rad_data_head));
    Rad_data_Qc = NaN(size(X2,1),numel(Rad_data_head));
    for l = 1:numel(Rad_data_head)
        Rad_data(:,l) = X2(:,ismember(X2_head,Rad_data_head(l)));
        Rad_data_Qc(:,l) = X2qc(:,ismember(X2_head,Rad_data_head(l)));
    end   

    % Create the header of the variables used as drivers     
    prov_d_head_switchoff={'SW_IN_f'}; % for the product with LUE
    prov_d_head_reco={'WS_f';'cos_wd';'sin_wd';'Tair_f';'TS_f';'SWC_f';'cos_doy';'sin_doy';'NeeNightime';}; % to estimate RECO
    prov_d_head_photo={'WS_f';'cos_wd';'sin_wd';'Tair_f';'SWC_f';'RadDaily';'RadDailySeas';'SW_IN_POT';'SW_IN_f';'dSW_IN_POT';'VPD_f';'GPPDailyprov';}; % to estimate LUE
     
    OutReco_NNcust = NaN(size(X2,1),5);
    OutGPP_NNcust = NaN(size(X2,1),5);
    OutNEE_tset_NNcust = NaN(size(X2,1),5);
    for m3 = 1:numel(uy)
        clear iPr
        iPr = find(TimeDate(:,1)==uy(m3));
        % cCopy the header of variables in a temporary object
        clear tX0_head tX1_head tX2_head
        tX2_head = X2_head;
        % Pick-up data for the year that we need to process
        clear tX0 tX1 tX2
        tX2 = X2(iPr,:);
        % also the quality check
        clear tX0qc tX1qc tX2qc
        tX2qc = X2qc(iPr,:);
        % Pick up the target variables and radiation data for the
        % year that we need to process
        clear tYtr tRad_data tmonth
        tYtr = Ytr(iPr,:);
        tRad_data = Rad_data(iPr,1:2);
        tNEE_fnet = NEE_fnet(iPr,1);
        % create a matrix to normalize data following the min_max criteria
        % x_norm = 2*((x-min)/(max-min)-0.5); where min = -(max(abs(x)) and max = max(abs(x))
        Xquad = [tX2 tNEE_fnet; -tX2 -tNEE_fnet];
        t1 = nanmin([tX2 tNEE_fnet],[],1);
        t2 = nanmax([tX2 tNEE_fnet],[],1);
        t1(1,1:end-1)=0;
        t2(1,1:end-1)=0;
        c_sat=1.15.*(abs(t1(end))+abs(t2(end)));
        t1(end)=-c_sat;
        t2(end)=+c_sat;
        Xquad=[Xquad; t1; t2];
                
        clear Inp_nor minInp maxInp Out_nor minOut maxOut
        [Inp_nor, minInp, maxInp, Out_nor,minOut, maxOut] = premnmx (Xquad(:,1:end-1)',Xquad(:,end)');
        i1 = [14 16 17 10 13 5 6 7 8 9 11 18];
        i2 = [14 16 17 10 12 13 3 4 19];
        i3 = 8;
        % at follow we create the customized NNc-part by setting the layers
        % number of inputs layers to be connected with hidden layers, total number
        % of hidden layers, connection among layers and so on.
        for i=1:5
            layer = eval(['SelectedModelsStruct.y', num2str(uy(m3)), '.ALL.rep(i).CANN.LW']);
            dim1 = size(layer{2,1}, 2);
            dim2 = size(layer{4,3},2);
            input = eval(['SelectedModelsStruct.y', num2str(uy(m3)), '.ALL.rep(i).CANN.IW']);
            Inp_nor = Inp_nor(:,1:size(tX2,1));
            Out_nor = Out_nor(:,1:size(tX2,1));

            clear ifg
            ifg = find(isfinite(prod([Inp_nor; Out_nor],1)));
            ifg = randsample(ifg,100); 
            clear net
            net = network;
            net.numInputs = 3;
            net.numLayers = 6;
            net.inputConnect = [1 0 0; 0 0 0; 0 1 0; 0 0 0; 0 0 1; 0 0 0;]; % [1 0 0; 0 1 0; 0 0 1; 0 0 0;];
            net.layerConnect = [0 0 0 0 0 0; 1 0 0 0 0 0; 0 0 0 0 0 0; 0 0 1 0 0 0; 0 1 0 0 0 0; 0 0 0 1 1 0;]; %  [0 0 0 0; 1 0 0 0; 1 0 0 0; 0 1 1 0;];
            net.biasConnect = [1; 1; 1; 1; 1; 0]; %  1 0 0 0; 0 0 0 0; 0 1 1 0];
            net.outputConnect = [0 0 0 0 0 1];
            % Layers 1 and layers 2 are used to estimate LUE, the output of layer 2 (LUE)
            % enters as input in layer 5 to calculate GPP as the product LUE*SWIN
            % Layers 3 and layers 4 are used to estimate RECO;
            % The output of layer 4 ad 5 enter as inpput in layer 6 to calculate NEE as
            % the difference NEE = RECO-GPP
            % for each layer we set size (number of neurons), transfer function, initialization function, neuron operation on weighted input (sum or product)
            % layers for LUE
            net.layers{1}.size = dim1;
            net.layers{1}.transferFcn = 'tansig';
            net.layers{1}.initFcn = 'initnw';
            net.layers{1}.netInputFcn = 'netsum';
            % Here there is the logsig transferfunction to have positive outputs
            net.layers{2}.size = 1;
            net.layers{2}.transferFcn = 'logsig';
            net.layers{2}.initFcn = 'initnw';
            net.layers{2}.netInputFcn = 'netsum';
            % layers for RECO
            net.layers{3}.size = dim2;
            net.layers{3}.transferFcn = 'tansig';
            net.layers{3}.initFcn = 'initnw';
            net.layers{3}.netInputFcn = 'netsum';
            % Here there is the logsig transferfunction to have positive outputs
            net.layers{4}.size = 1;
            net.layers{4}.transferFcn = 'logsig';
            net.layers{4}.initFcn = 'initnw';
            net.layers{4}.netInputFcn = 'netsum';
            % in this layer we apply the product to calculate GPP = LUE*SWIN
            net.layers{5}.size = 1;
            net.layers{5}.transferFcn = 'poslin';
            net.layers{5}.initFcn = 'initnw';
            net.layers{5}.netInputFcn = 'netprod';
            % the following is the last node of the overall structure, were NEE is
            % calculated as NEE = RECO-GPP
            net.layers{6}.size = 1;
            net.layers{6}.transferFcn = 'purelin';
            net.layers{6}.initFcn = 'initnw';
            net.layers{6}.netInputFcn = 'netsum';
            % set also the cost function (mse), network initialization function and the training function
            net.performFcn = 'mse';
            net.initFcn = 'initlay';
            net.trainFcn = 'trainlm';
            % entering a subset of example input and output
            net.inputs{1}.exampleInput = Inp_nor(i1,ifg);
            net.inputs{2}.exampleInput = Inp_nor(i2,ifg);
            net.inputs{3}.exampleInput = Inp_nor(i3,ifg);
            net.outputs{1}.exampleOutput = Out_nor(ifg);
            clear ifg
            % finally initialize the network
            net=init(net);
            net.LW = layer;
            net.IW = input;
          
            output= net();




            
            Importance = permutationImportance(net)
            explainer = shapley(net);

        end 
    end
end