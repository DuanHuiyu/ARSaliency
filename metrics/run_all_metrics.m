function [AUC_Borji_score, AUC_Judd_score, sAUC_score, CC_score, IG_score, KL_score, NSS_score, SIM_score] ...
    = run_all_metrics(saliencyMap, fixationDensityMap, fixationMap, baselineMap)
% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% saliencyMap: saliencyMap computed by models;
% fixationDensityMap: visual attention map, (filtered fixationMap);
% fixationMap: fixation points map;
% baselineMap: other map, shuffled map, (all fixations from other images);

[AUC_Borji_score,tp,fp] = AUC_Borji(saliencyMap, fixationMap);
[AUC_Judd_score,tp,fp,allthreshes] = AUC_Judd(saliencyMap, fixationMap);
[sAUC_score,tp,fp] = AUC_shuffled(saliencyMap, fixationMap, baselineMap);
CC_score = CC(saliencyMap, fixationDensityMap);
IG_score = InfoGain(saliencyMap, fixationMap, baselineMap);
KL_score = KLdiv(saliencyMap, fixationMap);
NSS_score = NSS(saliencyMap, fixationMap);
SIM_score = similarity(saliencyMap, fixationDensityMap);

end