%% seg mask generation

clear
load('/media/outer/Data/swi_index1.mat');
base_dir = '/media/outer/Data/PD_cohort/MPRAGE/Registered/ROI';

rw_size = [512 512 180];
m_size = [512 512 180];
for i=64:64
    seg = zeros(m_size);
    
    SN = zeros(rw_size);
    RN = zeros(rw_size);
    DN = zeros(rw_size);
    
    GP = zeros(rw_size);
    Put = zeros(rw_size);
    Tha = zeros(rw_size);
    CN = zeros(rw_size);
       
    seg(seg==0) = 14;
    GM = load_untouch_nii([base_dir '/bilc1_' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    WM = load_untouch_nii([base_dir '/bilc2_' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    CSF = load_untouch_nii([base_dir '/bilc3_' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    bone = load_untouch_nii([base_dir '/bilc4_' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    mus = load_untouch_nii([base_dir '/bilc5_' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    air = load_untouch_nii([base_dir '/bilc6_' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    
    GMm = single(imresize3(GM.img,m_size));
    GMm = imbinarize(GMm, 0.5);   
    WMm = single(imresize3(WM.img,m_size));
    WMm = imbinarize(WMm, 0.5); 
    CSFm = single(imresize3(CSF.img,m_size));
    CSFm = imbinarize(CSFm, 0.5); 
    bonem = single(imresize3(bone.img,m_size));
    bonem = imbinarize(bonem,0.5);
    musm = single(imresize3(mus.img,m_size));
    airm = single(imresize3(air.img,m_size));
    musm = imbinarize(musm,0.5);   
    airm = imbinarize(airm,0.5);
    
    seg(GMm==1) = 9;
    seg(WMm==1) = 8;
    seg(CSFm==1) = 10;
    seg(bonem==1) = 13;
    seg(musm == 1) = 15;
    
    %loading vessel data
    ves = load_nii(['/media/outer/Data/PD_cohort/SWI/Origin/dicom2nii/vessel/qsm' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    ves = imresize3(ves.img,m_size);
    ves = imbinarize(ves,0.5);
    seg(ves==1) = 11;

    LGP = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/195-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    RGP = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/196-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    
    LPut = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/193-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    RPut = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/194-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    
    LTha = load_nii(['/media/outer/Data/ROImask/Left_Thalamus/hLTha' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '_thr.nii.gz']);
    RTha = load_nii(['/media/outer/Data/ROImask/Right_Thalamus/hRTha' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '_thr.nii.gz']);
    
    LCN = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/191-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    RCN = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/192-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    
    GPo = LGP.img+RGP.img;
    Puto = LPut.img + RPut.img;
    Thao = LTha.img + RTha.img;
    CNo = LCN.img + RCN.img;
    
    GP(91:422,51:462,:) = GPo;
    Put(91:422,51:462,:) = Puto;
    Tha = Thao; 
    CN(91:422,51:462,:) = CNo;
    
    seg(GP==1)= 2;  %globus pallidus
    seg(Put==1)= 3;  %putamen
    seg(Tha==1)= 7;  %thalamus
    seg(CN==1)= 1;  %caudate nucleus
    
    LSN1 = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/197-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    RSN1 = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/198-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    LSN2 = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/199-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    RSN2 = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/200-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    
    LRN = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/201-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    RRN = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/202-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    
    LDN = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/203-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    RDN = load_nii(['/media/outer/Data/QSMmask/qsm_roi_bi/204-' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);

    SNo = LSN1.img+LSN2.img+RSN1.img+RSN2.img;
    RNo = LRN.img+RRN.img;
    DNo = LDN.img+RDN.img;
    
    SN(91:422,51:462,:) = SNo;
    RN(91:422,51:462,:) = RNo;
    DN(91:422,51:462,:) = DNo;
    
    SNm = imresize3(SN,m_size);
    SNm = imbinarize(SNm,0.5);
    
    RNm = imresize3(RN,m_size);
    RNm = imbinarize(RNm,0.5);
    
    DNm = imresize3(DN,m_size);
    DNm = imbinarize(DNm,0.5);
    
    seg(SNm == 1)= 6;
    seg(RNm == 1)= 4;
    seg(DNm == 1)= 5; 
    
    save_nii(make_nii(seg),['/media/outer/Data/simu_RC2/Raw_size/seg_map/seg' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
end


%% brain phantom genearation
clear
% using in vivo data for generating phantom
load('/media/outer/Data/swi_index1.mat');
base_dir = '/media/outer/Data/PD_cohort/';

for i=1:16
    ModelParams.R1map_file = [base_dir 'MPRAGE/Registered/T1R1/R1' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];
    ModelParams.R2starmap_file = [base_dir 'T2R2/R2s' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];
    ModelParams.M0map_file = [base_dir 'PD/rPD_' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];
    ModelParams.Segmentation_file = ['/media/outer/Data/simu_RC2/Raw_size/seg_map/seg' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];

    ModelParams.rawField_file = [base_dir 'SWI/Origin/Phase/phase' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];
    RealFreqMap=load_nii(ModelParams.rawField_file);
    
    [fieldgradient(:,:,:,1), fieldgradient(:,:,:,2), fieldgradient(:,:,:,3)]=gradient_3D(RealFreqMap.img,[],0);
    fieldgradient=(sos(fieldgradient,4));
    fieldgradient(fieldgradient>0.2)=1;
    fieldgradient(fieldgradient<=0.2)=0;
    save_nii(make_nii(fieldgradient),['/media/outer/Data/simu_RC2/Raw_size/hgrad_msk/msk' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']);
    
    ModelParams.BrainMask_file = [base_dir 'SWI/Origin/dicom2nii/mag/mag' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '_brain_mask.nii.gz'];

    % file that defines regions where R2* values can and can't be trusted
    ModelParams.highGradMask_file = ['/media/outer/Data/simu_RC2/Raw_size/hgrad_msk/msk' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];

    % File with the various parameters to create a susceptbility map
    % ModelParams.ChiModulation_file = 'data/chimodel/paramaters.mat';
    ModelParams.ChiModulation_file = '/media/outer/Data/DSC_3015069.02_542_v1/data/chimodel/paramatersCalcificationFree.mat';
%     ModelParams.ChiModulation_file = '/media/outer/Data/DSC_3015069.02_542_v1/data/chimodel/paramaters.mat';

    ModelParams.OutputChiModel_file = ['/media/outer/Data/simu_RC2/Raw_size/Chi/2LBchi' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];

    %     CreateOwnRealisticPhantom(ModelParams)
    params = ModelParams;
    load(params.ChiModulation_file,'label'); % this has the information on the reference susceptibility in each region

    for k=1:length(label)
        R12chi(k)=label(k).R12chi;
        R1only2chi(k)=label(k).R1only2chi;
        R2star2chi(k)=label(k).R2star2chi;
%         label(k).chiref = normrnd(label(k).chiref,0.01,1);
        if k > 12
            label(k).chiref = normrnd(label(k).chiref/2,0.01,1);
        else
            label(k).chiref = normrnd(label(k).chiref,0.01,1);  
        end
    end


    FinalSegment = load_nii(params.Segmentation_file); % this has the infromation
    FinalSegm = FinalSegment.img;
    
    mask = load_untouch_nii(params.BrainMask_file);
    mask.img = single(mask.img);
    
    R1 = load_nii(params.R1map_file);
    R1.img(isinf(R1.img))=0;
%     R1.img(mask.img==0)=-1;
%     R1filt = medfilt3(R1.img,[3 3 3]);
%     R1.img(R1.img==0 & mask.img==1)=R1filt(R1.img==0 & mask.img==1);
    
    tmp=Wavedec3Denoising(R1.img,25/1000,8,'db2','verysoft',double(R1.img~=0));
    R1.img = abs(real(tmp)); % same as for segmentation purposes

    R2star= load_nii(params.R2starmap_file);
    R2star.img(isinf(R2star.img))=0;
%     R2starfilt= medfilt3(R2star.img,[3 3 3]);
%     R2star.img(R2star.img==0 & mask.img==1) = R2starfilt(R2star.img==0 & mask.img==1);
%     save_nii(make_nii(R2star.img),params.OutputChiModel_file);
    
    tmp=Wavedec3Denoising(R2star.img,45/10,8,'db2','soft',double(R1.img~=0)); % same as for segmentation purposes
    R2star.img = abs(real(tmp ));

    M0=load_untouch_nii(params.M0map_file);

    HighGradMask=load_nii(params.highGradMask_file);
    RealFreqMap=load_nii(params.rawField_file);
    [fieldgradient(:,:,:,1), fieldgradient(:,:,:,2), fieldgradient(:,:,:,3)]=gradient_3D(RealFreqMap.img,[],0);
    fieldgradient=(sos(fieldgradient,4));
    fieldgradient(FinalSegm>=11)=0;

    fieldgradient=fieldgradient.*single(HighGradMask.img);

    %     %define acceptable threshold ???
    thresholdgrad=prctile(fieldgradient(fieldgradient~=0),[10 90]); % is this a good threshold, check on the fv of the fiedlgradient
    % looking at the figure 0.05 seems to be the gradient close to the edges
    % looking at the figure 0.2 seems to be a too large gradient

    fieldgradient(and(fieldgradient~=0,fieldgradient>0.2))=0.2;
    fieldgradient(and(fieldgradient~=0,fieldgradient<0.05))=0.05;
    fieldgradient(fieldgradient~=0)=(fieldgradient(fieldgradient~=0)-0.05)/(0.2-0.05);
    WeightingBetweenModels=cos(fieldgradient);


    Chimap3=0*zeros(size(FinalSegm));% in this one the modulation on the regions of high gradients is only based on the R1 maps
    Chimap4=0*zeros(size(FinalSegm));% in this one the modulation is only based on the R1 maps
    FWHM_V=1.2;
    %HG_mask_index=find(HighGradMask.img==1);
    %LG_mask_index=find(HighGradMask.img==0);
    HG_mask_index=[];
    LG_mask_index=find(or(HighGradMask.img==0,HighGradMask.img==1));


    ProbabilityAccumulated=0*zeros(size(FinalSegm));

    for k=1:length(label)

        Mask=FinalSegm==k;
        indexes=find(FinalSegm==k);
        indexes_HG=[];
        %    indexes=find(and(FinalSegm==k,HighGradMask.img==0));
        %    indexes_HG=find(and(FinalSegm==k,HighGradMask.img==1));

        if sum(Mask(:))>1

            Mask_smooth=real(smooth3D(double(Mask),FWHM_V,[1,1,1]));
            %         Chimap2=Chimap2+Mask_smooth.*(label(k).chiref+...
            %             R2star2chi(k)*(R2star.img-mean(R2star.img(indexes)))+...
            %             R12chi(k)*(R1.img-mean(R1.img(indexes))));
            % ensures the modulation is only applied in regions of low gradient
            if k <= 10
                ProbabilityAccumulated=ProbabilityAccumulated+Mask_smooth;
                chitemp=...
                    R2star2chi(k)*(R2star.img(LG_mask_index)-mean(R2star.img(indexes)))+...
                    R12chi(k)*(R1.img(LG_mask_index)-mean(R1.img(indexes)));

                % trying to find a way to avoid creating outlyers because of errors on the R2* and R1 maps...
                NSTD=3;
                smoothThresh=0.9; % region where the width of accepted values is calculated
                smoothThresh2=0.05; % region where the width of accepted values is enforced
                modulation_std = std(chitemp(Mask_smooth(LG_mask_index)>smoothThresh));
                modulation_mean = mean(chitemp(Mask_smooth(LG_mask_index)>smoothThresh));
                % makes all the outliers have the average value of the segmented tissue
                chitemp(and(Mask_smooth(LG_mask_index)>smoothThresh2,abs(chitemp-modulation_mean)>NSTD*modulation_std))=modulation_mean;


                Chimap3(LG_mask_index)=Chimap3(LG_mask_index)+Mask_smooth(LG_mask_index)...
                    .*(label(k).chiref+1*chitemp);

                % does the all R1 based method

                chitemp=...
                    R1only2chi(k)*(R1.img(LG_mask_index)-mean(R1.img(indexes)));

                % trying to find a way to avoid creating outlyers because of errors on the R2* and R1 maps...
                modulation_std = std(chitemp(Mask_smooth(LG_mask_index)>smoothThresh));
                modulation_mean = mean(chitemp(Mask_smooth(LG_mask_index)>smoothThresh));
                %             chitemp(and(Mask_smooth(LG_mask_index)>smoothThresh2,chitemp>modulation_mean+NSTD*modulation_std))=modulation_mean+NSTD*modulation_std;
                %             chitemp(and(Mask_smooth(LG_mask_index)>smoothThresh2,chitemp<modulation_mean-NSTD*modulation_std))=modulation_mean-NSTD*modulation_std;
                chitemp(and(Mask_smooth(LG_mask_index)>smoothThresh2,abs(chitemp-modulation_mean)>NSTD*modulation_std))=modulation_mean;
                Chimap4(LG_mask_index)=Chimap4(LG_mask_index)+Mask_smooth(LG_mask_index)...
                    .*(label(k).chiref+chitemp);

            else
                %    this is not quite correct because there will be some suseptibilitz
                %    missing in the regions arround vessels, dura and so on... hopefullz it
                %    is a small effect
                chitemp=R2star2chi(k)*(R2star.img(indexes)-mean(R2star.img(indexes)))+...
                    R12chi(k)*(R1.img(indexes)-mean(R1.img(indexes)));

                modulation_std = std(chitemp(Mask(indexes)>smoothThresh));
                modulation_mean = mean(chitemp(Mask(indexes)>smoothThresh));
                %             chitemp(and(Mask(indexes)>smoothThresh2,chitemp>modulation_mean+NSTD*modulation_std))=modulation_mean+NSTD*modulation_std;
                %             chitemp(and(Mask(indexes)>smoothThresh2,chitemp<modulation_mean+NSTD*modulation_std))=modulation_mean-NSTD*modulation_std;
                chitemp(and(Mask(indexes)>smoothThresh2,chitemp>modulation_mean+NSTD*modulation_std))=modulation_mean;
                chitemp(and(Mask(indexes)>smoothThresh2,chitemp<modulation_mean+NSTD*modulation_std))=modulation_mean;


                Chimap3(indexes)=Chimap3(indexes)+(Mask(indexes)-ProbabilityAccumulated(indexes))...
                    .*(label(k).chiref+1*chitemp);

                % only based on R1
                chitemp=...
                    R1only2chi(k)*(R1.img(indexes)-mean(R1.img(indexes)));

                modulation_std = std(chitemp(Mask(indexes)>smoothThresh));
                modulation_mean = mean(chitemp(Mask(indexes)>smoothThresh));
                %              chitemp(and(Mask(indexes)>smoothThresh2,chitemp>modulation_mean+NSTD*modulation_std))=modulation_mean+NSTD*modulation_std;
                %              chitemp(and(Mask(indexes)>smoothThresh2,chitemp<modulation_mean+NSTD*modulation_std))=modulation_mean-NSTD*modulation_std;
                chitemp(and(Mask(indexes)>smoothThresh2,chitemp>modulation_mean+NSTD*modulation_std))=modulation_mean;
                chitemp(and(Mask(indexes)>smoothThresh2,chitemp<modulation_mean+NSTD*modulation_std))=modulation_mean;
                Chimap4(indexes)=Chimap4(indexes)+(Mask(indexes)-ProbabilityAccumulated(indexes))...
                    .*(label(k).chiref+chitemp);


            end;
        end
    end;

    Chimap5=WeightingBetweenModels.^4.*Chimap3+(1-WeightingBetweenModels.^4).*Chimap4;

%     amsk = ones(size(mask.img));
%     amsk(FinalSegm>12)=0;
    chimodel=R1;
%     chimodel.img=Chimap5.*amsk;
    chimodel.img=Chimap5;
    save_nii(chimodel,params.OutputChiModel_file);
end

%% data simulation
clear

load('/media/outer/Data/swi_index1.mat');
SimulateData = 1;
% This are the model parameters
Protocol = 1 ;


    
base_dir = '/media/outer/Data/';
SeqParams{Protocol}.TR = [28e-3];                               % Repetition time in secs
SeqParams{Protocol}.TE = [23e-3];     % Echo time in secs
SeqParams{Protocol}.FlipAngle = 15;                             % flip angle in degrees above ernst angle (13) to improve gre contrast between Grey and white matter


SimParams{Protocol}.B0 = 3;
SimParams{Protocol}.B0_dir = [0 0 1];
SimParams{Protocol}.PhaseOffset = 1 ;                            % multiplier term of a quadratic phase over the brain
% 0 no phase offset; pi phase difference inside brain mask
SimParams{Protocol}.Shimm = 1;                                  % boolean 0 if no additional shimms are applied 1 if
% SimParams{Protocol}.Res = [0.6 0.6 1.3];                              %  Resolution of output    
SimParams{Protocol}.Res = [1 1 1];                              %  Resolution of output    

for i = 1:16
    
   
    ModelParams.Chimap_file = [base_dir 'simu_RC2/Raw_size/Chi/chi' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz']; % the model used in this paper has some minor differences in respect to those used in the challenge
    % ModelParams.Chimap_file = 'data/chimodel/ChiModelMIX.nii.gz'; % Sim2 of
    % Challenge
    % ModelParams.Chimap_file = 'data/chimodel/ChiModelMIX_noCalc.nii.gz'; Sim1
    % of Challenge

    ModelParams.R1map_file = [base_dir 'PD_cohort/MPRAGE/Registered/R1' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];
    ModelParams.R2starmap_file = [base_dir 'PD_cohort/T2R2/R2s' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];
    ModelParams.M0map_file = [base_dir 'PD_cohort/PD/rPD_' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];
    ModelParams.Segmentation_file = [base_dir 'simu_RC2/Raw_size/seg_map/seg' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];
    
    ModelParams.BrainMask_file =  [base_dir 'PD_cohort/SWI/Origin/dicom2nii/mag/mag' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '_brain_mask.nii.gz'];
    ModelParams.FA =  [base_dir 'PD_cohort/DTI/DTI_data/rDTI_' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '_FA.nii.gz'];
    ModelParams.V1 =  [base_dir 'PD_cohort/DTI/DTI_data/rDTI_' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '_V1.nii.gz'];

    SimParams{Protocol}.Output_dir = [base_dir 'simu_RC2/Raw_size/simudata/nobias' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2))]; % Output Directory

    if SimulateData == 1
        SimulatedData(ModelParams,SeqParams{Protocol},SimParams{Protocol})
    end

%         save( [ modelname, '/SimulationParameters.mat'],'ModelParams','SeqParams','SimParams')
end

function SimulatedData(ModelParams,SeqParams,SimParams)
    % 2nd order shimms are applied inside brain
    % mask. this is not done for the Brain
    % extracted output
    Sufix(1).name='BrainExtracted';
    Sufix(2).name='';
    savedata = 1;

    % some relevant parameters for a simulation
    B0=3;
    B0_dir=[0 0 1];
    padsize = [6 6 6];
    TE = 23;
    gyro = 42.57747892;% MHz/T    
    modelname = SimParams.Output_dir ;

    Folder=[modelname,'/GroundTruthHR/'];

        for BackGroundFieldRemoval = [2]
            Brain=load_untouch_nii(ModelParams.BrainMask_file);
            Brain=single(Brain.img);
            
            M0=load_untouch_nii(ModelParams.M0map_file);
            R1=load_untouch_nii(ModelParams.R1map_file);
            R1.img(isinf(R1.img)) = 0;
            
            R2star=load_untouch_nii(ModelParams.R2starmap_file);
            R2star.img(isinf(R2star.img)) = 0;
            
            R1filt = medfilt3(R1.img,[3 3 3]);
            R1.img(R1.img==0 & Brain==1)=R1filt(R1.img==0 & Brain==1);  

            R2starfilt= medfilt3(R2star.img,[3 3 3]);
            R2star.img(R2star.img==0 & Brain==1) = R2starfilt(R2star.img==0 & Brain==1);
            
            chi0=load_untouch_nii(ModelParams.Chimap_file);
            FinalSegment=load_untouch_nii(ModelParams.Segmentation_file);

            voxel_size = round(M0.hdr.dime.pixdim(2:4)*100)/100;
            
            FA=load_untouch_nii(ModelParams.FA);
            V1=load_untouch_nii(ModelParams.V1);

            chif = medfilt3(chi0.img,[3 3 3]);
            chi0.img(chi0.img>1 & Brain==1)=chif(chi0.img>1 & Brain==1);
            chi0.img(chi0.img>0.5 & Brain==1) = 0.5;
            chi0.img(chi0.img<-0.5 & Brain==1) = -0.5;
            chi = chi0 ;
            Brain(chi.img ==-0.5 & Brain==1) = 0;

            ChemShift=-3;
            WMmask=or(FinalSegment.img==8,FinalSegment.img==7);
            Microstructure=-5*(( ((sum(V1.img(:,:,:,1:2).^2,4))./(sum(V1.img.^2,4)))).^2-2/3).*(FA.img)/0.59.*(WMmask)+ChemShift*(WMmask);
            Microstructure(isnan(Microstructure))=0;
            Microstructure(isinf(Microstructure))=0;
            
            micphs = angle(exp(1i*2*pi*Microstructure*TE(end)-1i*pi/4));
            mic = QSM_star(micphs,WMmask,'TE',TE,'B0',B0,'H',B0_dir,'padsize',padsize,'voxelsize',voxel_size);
            chi.img = chi.img + mic/(2*pi*3*gyro*0.023)/1000;
            
%             mic = QSM_star(Microstructure,WMmask,'TE',TE,'B0',B0,'H',B0_dir,'padsize',padsize,'voxelsize',voxel_size);
%             chi.img = chi.img + mic/(2*pi*3*gyro*0.023)/1000;
%             m = mean(chi.img(Brain==1));
%             s = std(chi.img(Brain==1));
%             chi.img(Brain==1)=(chi.img(Brain==1)-m)./s;
%             chi.img(Brain==1)=reshape(zscore(chi.img(Brain==1)(:)),size(chi.img,1),size(chi.img,2),size(chi.img,3));
            
            % 'BrainExtracted';
            if BackGroundFieldRemoval == 1
                chi.img = (chi.img - mean(chi.img(Brain == 1))) .* Brain ;
                %             fv(Brain)
                %             fv(chi.img)
                Shimm = 0 ; % there is never a Bo shimming being applied when the the simulation is brain only
            end

            if BackGroundFieldRemoval == 2
                Shimm = SimParams.Shimm ;
            end
            
            % creates dipole Kernel
            dims = size(M0.img);
            %         D = create_dipole_kernel([0 0 1], voxel_size, dims, 1);
            %         field= real(ifftn(fftn(chi.img).*(D)))  ;% if chi is in ppm than field is in ppm
            %       doing some padding
            D = create_dipole_kernel(B0_dir , voxel_size, 2 * dims, 1);

            chitemp = ones( 2 * dims) * chi.img(end,end,end);
            chitemp (1:dims(1),1:dims(2),1:dims(3)) = chi.img;
            field= real(ifftn(fftn(chitemp).*(D)))  ;% if chi is in ppm then field is in ppm

            field = field(1:dims(1),1:dims(2),1:dims(3));
            clear chitemp
            clear D


            % Brain shimming

            if Shimm==0
                fieldb.img=field;
            else
                [~,fieldb,~]=PolyFitShimLike(make_nii(field),make_nii(single(Brain)),2);
            end
            
            % Brain Phase Offset

            if SimParams.PhaseOffset==0 % 
                PhaseOffset = 0;
            else
                [c , w ] = centerofmass(M0.img);
                [y,x,z] = meshgrid([1:dims(2)]-c(2), [1:dims(1)]-c(1), [1:dims(3)]-c(3));
                temp = (x/w(1)).^2 + (y/w(2)).^2 + (z/w(3)).^2 ;
                PhaseOffset = - temp/(max(temp(Brain==1))-min(temp(Brain==1)))*pi*SimParams.PhaseOffset;
            end


%             try
%                 figureJ(1),set(gcf,'Position',[1,29,1920,977])
%                 subplot(221)
%                 Orthoview(Brain)
%                 %    title('shim region')
%                 %    subplot(222)
%                 %   Orthoview(fieldb.img)
%                 subplot(222)
%                 Orthoview(fieldb.img,[],[-.01 .01])
%                 title('shimmed field')
%  
%                 %    subplot(224)
%                 %    Orthoview(field)
%                 subplot(224)
%                 Orthoview(field,[],[-.01 .01])
%                 title('unshimmed field')
%                 subplot(223)
%                 Orthoview(angle(exp(i*PhaseOffset)));
%                 title('B1 Phase Offset')
% 
%                 if savedata == 1
%                     savefig([Folder,'ShimmingAndB1phase',Sufix(BackGroundFieldRemoval).name])
%                 end
%             catch
%                 display('error displaying results');
%             end
            field = fieldb.img;
            clear fieldb


            %             for res = SeqParams.res
%             SeqParams = SeqParams{Protocol};
            for TE = SeqParams.TE
                
                if savedata == 1
                    %                 if model==1
                    FolderData=[modelname,'/Simulated_',num2str(floor(mean(SimParams.Res))),'p',num2str(floor(10*(mean(SimParams.Res)-floor(mean(SimParams.Res))))),'mm/']
                    FolderHR=[modelname,'/SimulatedHR/']
                    %                 else
                    %                     FolderData=['ChallengeDataOverModulated/Simulated_',num2str(floor(res)),'p',num2str(floor(10*(res-floor(res)))),'mm/']
                    %                     FolderHR=['ChallengeDataOverModulated/SimulatedHR/']
                    %
                    %                 end;

                    if exist(FolderData,'dir')~=7
                        mkdir(FolderData)
                    end

                    if exist(FolderHR,'dir')~=7
                        mkdir(FolderHR)
                    end

                end
                % config is a structure with all sort of parameters
                SequenceParam.TR=SeqParams.TR;                     %repetition time (in seconds);
                SequenceParam.TE=TE;                     %echo time (in seconds);
                SequenceParam.theta=SeqParams.FlipAngle;                     %flip angle in degrees;
                if length(SimParams.Res)==1
                    SequenceParam.res=[1 1 1 ]*SimParams.Res;             %resolution in mm;
                elseif length(SimParams.Res)==3
                    SequenceParam.res=SimParams.Res;             %resolution in mm;
                else
                    display('you have not entered the resolution in the correct format, we [ 1 1 1 ] will be assumed')
                    SequenceParam.res=[1 1 1 ];             %resolution in mm;
                end
                TissueParam.M0=double(M0.img);              %Water Concentration (arbitrry Units);
                TissueParam.R1=double(R1.img);                      %longitudinal relaxation rate (1/s);
                TissueParam.R2star=R2star.img;                  %apparent transverse relaxation rate (1/s);
                TissueParam.field=field * B0 * gyro;                   %field should be (1/s); - this assumes the field was calculated in ppm
                %     this to have the info of the
                TissueParam.PhaseOffset = PhaseOffset;             %PhaseOfsett at TE=0;
                TissueParam.res = voxel_size       ;             %resolution in mm;

                [sig,vox_out,sigHR]= DataSimulation(SequenceParam,TissueParam);
                
                % micstructure
%                 ChemShift=-3;
%                 WMmask=or(FinalSegment.img==8,FinalSegment.img==7);
%                 Microstructure=-5*(( ((sum(V1.img(:,:,:,1:2).^2,4))./(sum(V1.img.^2,4)))).^2-2/3).*(FA.img)/0.59.*(WMmask)+ChemShift*(WMmask);
%                 Microstructure(isnan(Microstructure))=0;
%                 Microstructure(isinf(Microstructure))=0;
                
                % checking the complex data created at the two resolutions
%                 slicepercent=40 ; % this is only to create a data visualization
%                 dims_start=size(sigHR);
%                 dims_end=size(sig);
%                 figureJ(3)
%                 set(gcf,'Position',[1,29,1920,977])
%                 colormap(gray)
%                 subplot(2,2,1)
%                 imab(abs(sig(:,:,round(slicepercent*dims_end(3)/100)))),title('Low Res Magnitude')
%                 subplot(2,2,2)
%                 imab(angle(sig(:,:,round(slicepercent*dims_end(3)/100)))),title('Low Res Phase')
% 
%                 subplot(2,2,3)
%                 imab(abs(sigHR(:,:,round(slicepercent*dims_start(3)/100)))),title('High Res Magnitude')
%                 subplot(2,2,4)
%                 imab(angle(sigHR(:,:,round(slicepercent*dims_start(3)/100)))),title('High Res Phase')

                if savedata == 1
%                     savefig([FolderData,'HighvsLowRes_TE',num2str(TE*1000),Sufix(BackGroundFieldRemoval).name])
                    save_nii(make_nii(abs(sig),vox_out),[FolderData,'Magnitude_TE',num2str(TE*1000,'%03.f'),Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
                    save_nii(make_nii(angle(sig),vox_out),[FolderData,'Phase_TE',num2str(TE*1000,'%03.f'),Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
%                     save_nii(make_nii(angle(exp(1i*2*pi*Microstructure*TE(end)-1i*pi/4).*sig),vox_out),[FolderData,'MicPhase_TE',num2str(TE*1000,'%03.f'),Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
                
                end
                if savedata == 1
                    save_nii(make_nii(abs(sigHR),voxel_size),[FolderHR,'Magnitude_TE',num2str(TE*1000,'%03.f'),Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
                    save_nii(make_nii(angle(sigHR),voxel_size),[FolderHR,'Phase_TE',num2str(TE*1000,'%03.f'),Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
%                     save_nii(make_nii(angle(exp(1i*2*pi*Microstructure*TE(end)-1i*pi/4).*sigHR),vox_out),[FolderHR,'MicPhase_TE',num2str(TE*1000,'%03.f'),Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
                    save_nii(make_nii(chi.img.*Brain,voxel_size),[FolderHR,'Chi_opt',Sufix(BackGroundFieldRemoval).name,'.nii.gz']);
                    save_nii(make_nii(chi.img,voxel_size),[FolderHR,'Chi_all',Sufix(BackGroundFieldRemoval).name,'.nii.gz']);
                    save_nii(make_nii(Brain,voxel_size),[FolderHR,'msk_opt',Sufix(BackGroundFieldRemoval).name,'.nii.gz']);
                    save_nii(make_nii(TissueParam.field,voxel_size),[FolderHR,'phs_tissue',Sufix(BackGroundFieldRemoval).name,'.nii.gz']);
                end
%                 figureJ(3)
%                 set(gcf,'Position',[1,29,1920,977])
%                 colormap(gray)
%                 subplot(2,2,1)
%                 Orthoview(abs(sig)),title('Low Res Magnitude')
%                 subplot(2,2,2)
%                 Orthoview(angle(sig)),title('Low Res Phase')
% 
%                 subplot(2,2,3)
%                 Orthoview(abs(sigHR)),title('High Res Magnitude')
%                 subplot(2,2,4)
%                 Orthoview(angle(sigHR)),title('High Res Phase')

            end


            % checking the lowres susceptibility ground truth
            dims_start=size(sigHR);
            dims_end=size(sig);
            clear sig
%             figureJ(5)
%             set(gcf,'Position',[1,29,1920,977])
            % probably more correct - but does not match the data signal
            % simulation part
            % chi_end = real( ifftshift(ifftn(ifftshift(crop(fftshift(fftn(fftshift(chi.img.*Brain))),dims_end)))))*prod(dims_end)/prod(dims_start);
            % probably less correct - but does match the data signal
            % simulation part
             chi_end = real( (ifftn(ifftshift(crop(fftshift(fftn((chi.img))),dims_end)))))*prod(dims_end)/prod(dims_start);
            chi_end = permute(unring(permute(unring(chi_end),[3 1 2])),[2 3 1])	;
            [X,Y,Z] = ndgrid(single(1:dims_start(1)),single(1:dims_start(2)),single(1:dims_start(3)));
            [Xq,Yq,Zq] = ndgrid(single(linspace(1,dims_start(1),dims_end(1))),...
                single(linspace(1,dims_start(2),dims_end(2))),single(linspace(1,dims_start(3),dims_end(3))));

            chi_end_interp = interpn(X,Y,Z,single(chi.img.*Brain),Xq,Yq,Zq);



%             subplot(311)
%             Orthoview(chi.img.*Brain,[],[-0.04 0.08]);colorbar; % high res susceptibility map
%             title('High Res susceptibility')
%             subplot(312)
%             Orthoview(chi_end,[],[-0.04 0.08]);colorbar; % high res susceptibility map
%             title('Low Res susceptibility FFTcrop')
%             subplot(313)
%             Orthoview(chi_end_interp,[],[-0.04 0.08]);colorbar; % high res susceptibility map
%             title('Low Res susceptibility Interp')
            if savedata == 1
%                 savefig([FolderData,'GroundTruthSusceptibility',Sufix(BackGroundFieldRemoval).name])
                %             save_nii(make_nii(Brain,vox_out),[FolderData,'Brain',Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
                save_nii(make_nii(chi_end_interp,vox_out),[FolderData,'Chi_interp',Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
                save_nii(make_nii(chi_end,vox_out),[FolderData,'Chi_crop',Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
                %                 end
                clear chi_end
                clear chi_end_interp
                % testing the downsampling of the segmentation model
                % I would have probably prefered to do this segmentation  on the
                % probability maps used to create the ground truth
                % susceptibility...
                temp=zeros([size(Xq), max(FinalSegment.img(:))]);
                for label=1:max(FinalSegment.img(:))

                    temp(:,:,:,label) = interpn(X,Y,Z,single(FinalSegment.img==label),Xq,Yq,Zq);
                    %                 makes the computation really slow
                    %                 tmp = real( (ifftn(ifftshift(crop(fftshift(fftn(single(FinalSegment.img==label))),dims_end)))))*prod(dims_end)/prod(dims_start);
                    %                 temp(:,:,:,label) = permute(unring(permute(unring(tmp),[3 1 2])),[2 3 1])	;


                end
                [temp_val temp_pos]= max(temp,[],4);
                Segment_LR=temp_pos;
%                 subplot(211),Orthoview(Segment_LR)
%                 subplot(212),Orthoview(FinalSegment.img)
                temp= interpn(X,Y,Z,single(Brain),Xq,Yq,Zq);
                Brain_LR= temp>0.9;
                if savedata == 1
                    save_nii(make_nii(Segment_LR,vox_out),[FolderData,'FinalSegment',Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
                    save_nii(make_nii(single(Brain_LR),vox_out),[FolderData,'Brain',Sufix(BackGroundFieldRemoval).name,'.nii.gz'])
                end
                clear temp, clear temp_val, clear temp_pos, clear Segment_LR, clear Brain_LR


                % Checking how it compares to real data that is stored in
                % data/raw
                %
%                 try
%                     TE = [23]*10e-3;
% 
%                     files{1}.name = [base_dir 'SWI/Origin/dicom2nii/mag/mag' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];
%                     files{2}.name = [base_dir 'SWI/Origin/Phase/phase' num2str(swi_index(i,1)) '-' num2str(swi_index(i,2)) '.nii.gz'];
% 
%                     MagnData = load_untouch_nii([files{1}.name]);
%                     PhaseData = load_untouch_nii([files{2}.name]);
%                     figureJ(10)
%                     set(gcf,'Position',[1,29,1920,977])
%                     subplot(2,2,1)
%                     imab((MagnData.img(:,:,round(slicepercent*dims_start(3)/100)))),title('Real Magnitude')
%                     subplot(2,2,2)
%                     imab(PhaseData.img(:,:,round(slicepercent*dims_start(3)/100))),title('Real Phase')
%                     subplot(2,2,3)
%                     imab(abs(sigHR(:,:,round(slicepercent*dims_start(3)/100)))),title('High Res Magnitude')
%                     subplot(2,2,4)
%                     imab(angle(sigHR(:,:,round(slicepercent*dims_start(3)/100)))),title('High Res Phase')
% 
%                     subplot(2,2,1)
%                     Orthoview((MagnData.img)),title('Real Magnitude')
%                     subplot(2,2,2)
%                     Orthoview(PhaseData.img),title('Real Phase')
%                     subplot(2,2,3)
%                     Orthoview(abs(sigHR)),title('High Res Magnitude')
%                     subplot(2,2,4)
%                     Orthoview(angle(sigHR)),title('High Res Phase')
%                     clear MagnData
%                     clear PhaseData
%                     clear sigHR
%                     if savedata == 1
%                         savefig([Folder,'ComparisonToRealData',Sufix(BackGroundFieldRemoval).name])
%                     end
%                 end
            end

        end
end

