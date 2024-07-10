###############################            DATA AUGUMENTATION FOR IMAGES         ###############################################

from scipy.ndimage.interpolation import shift

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)): #shifted 1px right, left, down and up
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented)) #to shuffle the dataset
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]



##################################  Displaying of images in subplots   #############################################################

fig, axes = plt.subplots(10, 10, figsize=(8, 8),subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),transform=ax.transAxes, color='green')
    


#########################################   Polynomial Regression using GridSearchCV ################################################
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs))

param_grid = {'polynomialfeatures__degree': np.arange(21)}

grid = GridSearchCV(PolynomialRegression(), param_grid)

grid.fit(X,y)
y_pred = grid.best_estimator_.predict(X_test)

plt.scatter(X,y)
plt.plot(X_test,y_pred)

#########################################     To Perform GridSearchCV      #########################################3
from sklearn.model_selection import GridSearchCV

model = RandomForestRegressor()

param_grid = [
{'n_estimators': [100,200], 'max_features': [8,10,12]}
]

grid = GridSearchCV(model, param_grid, cv=5,scoring='neg_mean_squared_error', return_train_score=True)

grid.fit(X_train,y_train)

#############################################  Best Model Selection   ###############################################################

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

model_params = {
    'svm':{
        'model': SVC(gamma='auto'),
        'params':{
            'C':[1,10,20],
            'kernel':['rbf', 'linear']
        }
    },
    
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[1,5,10]
        }
    },
    'logistic_regression':{
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params':{
            'C':[1,5,10]
        }
    }
}

scores = []

for model_name,mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train,y_train)
    scores.append({
        'model':model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
pd.DataFrame(scores, columns=['model','best_score','best_params'])

####################################################  SVC Visualizer ###########################################################
#The larger the C, the smaller the street, hence the margin is hard

from sklearn.svm import SVC
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a two-dimensional SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1],s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
model = SVC()
model.fit(X,y)

#For Linear SVC
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)

#For Radial SVC
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],s=300, lw=1, facecolors='none')


###########################################    K-MEANS CLUSTERING VISUALIZER  ########################################################
#Kmeans  is used for circular datasets 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
        
kmeans = KMeans(n_clusters=i, random_state=0)
plot_kmeans(kmeans, X2D)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', s=100 ,zorder=3)


###################################     GAUSSIAN MIXTURE CLUSTERING VISUALIZER       ##################################################
#Gaussian mixtures is used for both circular and non-circular datasets such as ellipse... it uses a probabilistic approach

from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
        
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
        
gmm = GaussianMixture(n_components=i, random_state=42)
plot_gmm(gmm, X2D)

##############################################   MODEL VISUALIZER    ####################################################################

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
    clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
    np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
    levels=np.arange(n_classes + 1) - 0.5,
    cmap=cmap, clim=(y.min(), y.max()),
    zorder=1)
    ax.set(xlim=xlim, ylim=ylim)
    
    
    
################################################## POS TAGGING MEANING ####################################################
NN: Noun, singular or mass
NNS: Noun, plural
NNP: Proper noun, singular
NNPS: Proper noun, plural
PRP: Personal pronoun
PRP$: Possessive pronoun
RB: Adverb
RBR: Adverb, comparative
RBS: Adverb, superlative
VB: Verb, base form
VBD: Verb, past tense
VBG: Verb, gerund or present participle
VBN: Verb, past participle
VBP: Verb, non-3rd person singular present
VBZ: Verb, 3rd person singular present
JJ: Adjective
JJR: Adjective, comparative
JJS: Adjective, superlative
IN: Preposition or subordinating conjunction
DT: Determiner
CC: Coordinating conjunction
UH: Interjection
CD: Cardinal number
EX: Existential 'there'
FW: Foreign word
LS: List item marker
MD: Modal
POS: Possessive ending
RP: Particle
SYM: Symbol
TO: 'to'
WDT: Wh-determiner
WP: Wh-pronoun
WP$: Possessive wh-pronoun
WRB: Wh-adverb
    
    
########################### Visualize the Proportions of Categorical Feature  ####################################################
def plot_bar_chart_with_percent_label(df, categorical_variable, if_sort=False, gap_label_bar=0, figsize=(9,6)):
    # prepare data
    plot_data = df[[categorical_variable]].value_counts().reset_index(name='count')
    plot_data['percent'] = plot_data['count']/plot_data['count'].sum()
    if if_sort:
        x_order = plot_data.sort_values(by=['percent'], ascending=False)[categorical_variable]
    else:
        x_order = plot_data.sort_values(by=[categorical_variable], ascending=True)[categorical_variable]
    # plot
    fig = plt.figure(figsize=figsize)
    ax = sns.barplot(data=plot_data, x=categorical_variable, y='percent', order=x_order)
    # add label
    for p in ax.patches:
        x = p.get_x() + p.get_width()/2
        y = p.get_height() + gap_label_bar
        ax.annotate(text='{:.2f}%'.format(p.get_height()*100), xy=(x, y), ha='center')
    ax.margins(y=0.1)
    
    

##################### Annotated Stacked bars for value_counts percentage(Binary prediction)  #############################
def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    """
    Add value annotations to the bars
    """

    # Iterate over the plotted rectanges/bars
    for p in ax.patches:
        
        # Calculate annotation
        value = str(round(p.get_height(),1))
        # If value is 0 do not annotate
        if value == '0.0':
            continue
        ax.annotate(
            value,
            ((p.get_x()+ p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
            color=colour,
            size=textsize
        )


#to plot the proporions based on the target variable
df_total=df[['id','target']].groupby('target').count()
df_percentage=df_total/df_total.sum()*100
ax=df_percentage.transpose().plot(kind='bar',stacked=True,figsize=(8,6),rot=0)
annotate_stacked_bars(ax, textsize=14)
plt.legend(['Yes','No'],loc='upper right')

### You can also categorize the target with two variables and find the percentage
#f1 is a categorical feature
#f2 could be any feature or id
df_new = df.groupby(['f1', 'target'])['f2'].count().unstack().fillna(0)
df_percentage = (df_new.div(df_new.sum(axis=1), axis=0)*100).sort_values(ascending=False,by=0)#the div divides each column element-wise by the total

ax=df_percentage.plot(kind='bar',stacked=True,figsize=(8,6),rot=75)
annotate_stacked_bars(ax, textsize=14)


#############################     Random Forest Classifier Feature Importance     ###############################################
# scaling and one hot encoding is not needed in random forest...just use LabelEncoder for categorical features

feature_importance_df = pd.DataFrame(data={'feature_name':X.columns, 'feature_importance':[0]*len(X.columns)})
feature_importance_df['feature_importance'] = feature_importance_df['feature_importance'] + (model.feature_importances_)

fig = plt.figure(figsize=(12,10))
ax = sns.barplot(data=feature_importance_df.sort_values(by=['feature_importance'], ascending=False), y='feature_name', x='feature_importance')


#####################################         FOLDER IMAGES DATASET PREPARATION                  ####################################
import cv2
import imghdr
import os

image_exts = ['jpeg','jpg', 'jfif', 'png', 'bmp']
data_dir = 'data'

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Isuue with image {}'.format(image_path))
            
data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()



############################################### TIMER for running a MODEL #####################################################
from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour,temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin,tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds' % (thour, tmin, round(tsec,2)))
        
        
start_time = timer()
scores = cross_val_score(LogisticRegression(),X,y,cv=5)
timer(start_time)
scores.mean()

