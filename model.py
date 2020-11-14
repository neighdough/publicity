import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from configparser import ConfigParser
from sqlalchemy import create_engine
import os
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
home = os.getenv("HOME")
proj_home = os.path.join(home, "Dropbox/phd/dissertation")
fig_home = os.path.join(home, proj_home, "figs")
sys.path.append(os.path.join(home, "dev"))
from disorientation import analysis

cnx_dir = os.getenv("CONNECTION_INFO")
parser = ConfigParser()
parser.read(os.path.join(cnx_dir, "db_conn.ini"))
psql_params = {k:v for k,v in parser._sections["disorientation"].items()}
psql_string = "postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(psql_string.format(**psql_params))
pd.set_option('display.width', 180)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 125)
df = pd.read_sql("select * from all_features", engine)
df.fillna(0, inplace=True)
x_vars = [col for col in df.columns if col not in 
            ['numpermit', 'numdemo', 'geoid10', 'wkb_geometry', 
             'scale_const', 'scale_demo', 'net']]
min_max_scale = lambda x: (x-x.min())/(x.max() - x.min())
std_scale = lambda x: (x-x.mean())/float(x.std())
df['scale_const'] = min_max_scale(df.numpermit)
df['scale_demo'] = min_max_scale(df.numdemo)
#permit column is actually net construction, but needs to be named permit
#to run correctly in fact_heatmap function
df['net'] = df.scale_const - df.scale_demo
RANDOM_STATE = 1416

def get_model_features(yname):
    y = df[yname]
    features = analysis.select_features(yname, True, .2)
    X = df[features["index"]]
    return (y, features, X)

def feature_importance(yname, num_records=None):
    y, features, X = get_model_features(yname)
    rfr = RandomForestRegressor(max_features=None, oob_score=True,
                                 random_state=RANDOM_STATE, n_estimators=225)
    rfr.fit(X, y)
    feat_imp = rfr.feature_importances_
    feat_imp = 100. * (feat_imp/feat_imp.max())

    #select and sort feature importances in descending order
    if num_records: #only select top n
        sorted_idx = np.argsort(feat_imp)[::-1][:num_records]
        imp_name = "feature_importance_{0}_top_{1}".format(yname, num_records)
        figsize = (17,11)
    else: #select all
        sorted_idx = np.argsort(feat_imp)[::-1]
        imp_name = "feature_importance_{0}".format(yname)
        figsize = (34,22)
   #pos = np.arange(sorted_idx.shape[0]) + .5
    df = pd.DataFrame({"variable": X.columns[sorted_idx],
                        "feat_imp": feat_imp[sorted_idx]})
    df.to_sql(imp_name, engine, if_exists="replace")
    if yname == "net":
        color = "Oranges_r"
        y_str = "Net Construction"
    elif yname == "scale_const":
        color = "Greens_r"
        y_str = "Scaled Construction"
    elif yname == "scale_demo":
        color = "Purples_r"
        y_str = "Scaled Demolition"

    f, ax = plt.subplots(figsize=figsize)
    f = sns.barplot(x="feat_imp", y="variable", data=df,
                    palette=color)
    f.set_yticklabels(df.variable, fontsize=14)
    f.set_xlabel("Feature Importance {}".format(y_str))
    f.set_ylabel("Column Name")
    f.figure.tight_layout(pad=6.)
#        f.fig.suptitle("Mean Feature Importance for {}".format(yname))
    plt.savefig(fig_home+"/"+imp_name+".png")

#yname = "scale_demo"#"scale_const"#"net"
def model_performance(yname):
    """Run model and generate plots to compare model performance. Creates the
         following plots:
        - Spearman
        - Pearson
        - OOB
        - Scatter plot actual vs predicted
    """
    y, features, X = get_model_features(yname)
    rfr = RandomForestRegressor(max_features=None, warm_start=True,
            oob_score=True, random_state=RANDOM_STATE)
    from scipy.stats import spearmanr, pearsonr
    min_estimators = 50
    max_estimators = 500
    rfr_error = OrderedDict()
    for i in range(min_estimators, max_estimators + 1):
        rfr.set_params(n_estimators=i)
        rfr.fit(X, y)
        oob = rfr.oob_score_
        y_pred = rfr.oob_prediction_
        sp = spearmanr(y, y_pred)
        pe = pearsonr(y, y_pred)
        feat_imp = rfr.feature_importances_
        rfr_error[i] = {'error':oob, 
                    'spearman': sp, 
                    'pearson': pe, 
                    'feat_imp': feat_imp}
        print(i, '\n\toob: ', oob, '\n\tspearman: ', sp.correlation)
        print('\tpearson: ', pe[0])
        print()

#*************************** Plots *************************************************

    if yname == "net":
        color = "orange"
        y_label = "Net Construction"
    elif yname == "scale_const":
        color = "green"
        y_label = "Scaled Construction Value"
    elif yname == "scale_demo":
        color = "purple"
        y_label = "Scaled Demolition Value"
    x = list(rfr_error.keys())
    y_error = [rfr_error[k]['error'] for k in rfr_error.keys()]
    y_sp = [rfr_error[k]['spearman'].correlation for k in rfr_error.keys()]
    y_pe = [rfr_error[k]['pearson'][0] for k in rfr_error.keys()]
    plt.figure(figsize=(12,8))
    plt.subplot(311)
    plt.plot(x, y_error, label="OOB Accuracy", color=color, linewidth=2.25)
    plt.ylabel("OOB", fontsize=16)
    plt.yticks(fontsize=12) 
    plt.tight_layout()
    plt.title("OOB Accuracy", fontsize=18) 
    plt.subplot(312)
    plt.plot(x, y_sp, label="Spearman's R", color=color, linewidth=2.25)
    #plt.xlabel("n_estimators")
    plt.ylabel("R", fontsize=16)
    plt.yticks(fontsize=12) 
    plt.tight_layout(pad=1.75)
    plt.title("Spearman's Rho", fontsize=18)
    plt.subplot(313)
    plt.plot(x, y_pe, color=color, linewidth=2.25)
    plt.ylabel("p", fontsize=16)
    plt.yticks(fontsize=12) 
    plt.xlabel("n_estimators")
    plt.tight_layout(pad=1.75)
    plt.title("Pearson's R", fontsize=18)
    plt.savefig(fig_home+"/model_performance_{}".format(yname))
    

    #Y actual vs Y predicted
    x = rfr.oob_prediction_
    m, b = np.polyfit(x, y, 1)
    avg_error = np.average(y_error)
#    x_label_pos = np.percentile(rfr.oob_prediction_, 40)
    #y_label_pos = np.percentile(y, 99)
    error_str = "Average OOB Accuracy\n{:.{prec}f}".format(avg_error, prec=3)
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.scatter(rfr.oob_prediction_, y, color=color, s=25)
    ax.annotate(error_str, xy=(0,0), xytext=(0.2, 0.8), 
                fontsize=16, ha="center", va="center", textcoords="axes fraction")
    #plt.figure(figsize=(12,8))
    #plt.scatter(rfr.oob_prediction_, y, color=color, s=25)
    #plt.text(, error_str, ha="center", va="center", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.yticks(fontsize=12)
    plt.xlabel('OOB Prediction', fontsize=16)
    plt.xticks(fontsize=12)
    plt.title('Y-Predicted vs Y-Actual', fontsize=18)
    plt.plot(x, m*x+ b, '-', color='black')
    plt.savefig(fig_home+"/prediction_vs_actual_{}".format(yname))    


pca = PCA(n_components=1)
#X = df[x_vars]
X_new = pca.fit_transform(X)
y_pos_idx = y[y >= 0].index
y_neg_idx = y[y < 0].index
plt.figure(figsize=(8,8))
if yname == "net":
    plt.scatter(X_new[y_pos_idx], y_pred[y_pos_idx], c='Green', s=30)
    plt.scatter(X_new[y_neg_idx], y_pred[y_neg_idx], c='Purple', s=30)
    plt.ylabel('Net Construction', fontsize=20)
    plt.xlabel('Principal Component', fontsize=20)
elif yname == "scale_const":
    plt.scatter(X_new, y_pred, c='Green', s=30)
    plt.xlabel("Principal Component", fontsize=20)
    plt.ylabel("Scaled Construction Permits", fontsize=20)
elif yname == "scale_demo":
    plt.scatter(X_new, y_pred, c='Purple', s=30)
    plt.ylabel("Scaled Demolition Permits", fontsize=20)
    plt.xlabel("Principal Component", fontsize=20)

plt.scatter(y, y_pred)

def poly_fit(x_label, y_label, order):
    new_df = df.copy()
    xp = new_df[df[y_label]>0][x_label]
    yp = new_df[df[y_label]>0][y_label]
    
    xn = new_df[df[y_label]<0][x_label]
    yn = new_df[df[y_label]<0][y_label]
   
    zp = np.polyfit(xp, yp, order)
    pp = np.poly1d(zp)
    lp = np.linspace(xp.min(), xp.max(), xp.shape[0])
    
    zn = np.polyfit(xn, yn, order)
    pn = np.poly1d(zn)
    ln = np.linspace(xn.min(), xn.max(), xn.shape[0])

    colors = lambda y: "Purple" if y < 0 else "Green"
    new_df["color"] = new_df[y_label].apply(colors)
    new_df.plot.scatter(x=x_label, y=y_label, color=new_df.color, s=8)
    plt.plot(lp, pp(lp), color="Green")
    plt.plot(ln, pn(ln), color="Purple")
    plt.savefig(fig_home+"/scatter_plot_with_fitline_{}.png".format(x_label))
