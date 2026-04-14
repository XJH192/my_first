import pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

rdf = pd.read_csv('result/RUN_LUCAS_Transformer_NoImage_D_2026_03_28_T_21_41_best.csv')
OC_MAX = 560.2

pid = 32421754
row = rdf[rdf['point_id'] == pid]
if len(row) > 0:
    r = row.iloc[0]
    real_oc = r['y_real'] * OC_MAX
    pred_oc = r['y_pred'] * OC_MAX
    print(f'point_id: {pid}')
    print(f'y_real (norm): {r["y_real"]:.6f}  -> OC: {real_oc:.2f} g/kg')
    print(f'y_pred (norm): {r["y_pred"]:.6f}  -> OC: {pred_oc:.2f} g/kg')
else:
    print('result file does not contain this point')
    print('total rows:', len(rdf))

# 整体误差分布
rdf['pred_oc'] = rdf['y_pred'] * OC_MAX
rdf['real_oc'] = rdf['y_real'] * OC_MAX
rdf['err'] = abs(rdf['pred_oc'] - rdf['real_oc'])
print('\n误差最大的前5个点:')
print(rdf.nlargest(5, 'err')[['point_id','real_oc','pred_oc','err']].to_string())
print('\n低OC(<20)样本统计:')
low = rdf[rdf['real_oc'] < 20]
print(f'  样本数: {len(low)}')
print(f'  平均真实OC: {low["real_oc"].mean():.2f}')
print(f'  平均预测OC: {low["pred_oc"].mean():.2f}')
print(f'  平均误差: {low["err"].mean():.2f}')
