from matplotlib import pyplot as plt
import pickle
import numpy as np
from torchvision.models import resnet18

def epoch_mean(array, window):

    mean = np.zeros(array.size//window)

    for i in range(array.size//window):
        mean[i]= np.mean(array[i*window:i*window+window])

    return np.array(mean)




with open('/home/davidetalon/Desktop/transfer-learning/saved_metrics/15-07-2020_15-50-48/bincifar_metrics.pkl', 'rb') as handle:
    data = pickle.load(handle)

metrics = data['run_metrics']
runs = data['runs']
losses = np.array(data['losses']) #(run, type of loss, measurement)
print(losses.shape)

plt.plot(epoch_mean(losses[0, 0, :]/losses[0, 0, :].max(), 29), label='fitting loss')
plt.plot(epoch_mean(losses[0, 1, :],29),  label='corr')
plt.plot(epoch_mean(losses[0, 2, :]/losses[0, 2, :].max(), 29), label='recon loss')
plt.title('Normalized losses')
plt.ylabel('Loss')
plt.xlabel('Epochs')


plt.legend()
plt.show()

# raise SystemError(0)
# (run, measurement, metric, task)

# print(runs)

# train_metrics = data['train_metrics']
metrics = np.array(metrics)


with open('/home/davidetalon/Desktop/transfer-learning/saved_metrics/04-07-2020_21-31-37/bincifar_metrics.pkl', 'rb') as handle:
    data_2 = pickle.load(handle)
# metrics_2 = data_2['run_metrics']
# metrics_2 = np.array(metrics_2)

##########################################
# print(runs)
plt.plot(metrics[0, :, -2, 1], label='STL')
plt.plot(metrics[1, :, -2, 1], label='Corr')
plt.title('T2: Validation losses during fine-tuning')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')

plt.legend()
plt.show()

raise SystemError(0)
# plt.plot(metrics[1, :, -2, 0], color='C1', label='BN+FC')
# plt.plot(metrics[1, :, 0, 0], linestyle='dashed', color='C1')
# plt.plot(metrics[2, :, -2, 0], color='C2', label='BN')
# plt.plot(metrics[2, :, 0, 0], linestyle='dashed', color='C2')
# plt.plot(metrics[3, :, -2, 0], color='C3', label='FC')
# plt.plot(metrics[3, :, 0, 0], linestyle='dashed', color='C3')

# plt.title('T1: Parameters capacity')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.legend()
# plt.show()
##############################################

# for run in range(metrics.shape[0]):
#     plt.plot(metrics[run, :, -2, 0], label='lr='+str(runs[run]['lr']))
# plt.title('T1: Validation accuracy vs learning rate')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.legend()
# plt.show()
#############################################



# for run_idx in range(40, 50):
#     print(run_idx, runs[run_idx])
#     plt.plot(metrics[run_idx, :, -2, 0], label=run_idx)
# plt.legend(loc='lower right')
# plt.title('T2:Entire network training with encoder')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.show()


# plt.plot(train_metrics[0, -1, :], label = 'test')
# plt.plot(train_metrics[0, 0, :], label ='train')
# plt.show()
correlations = np.array(data['correlations'])
print(correlations.shape)
for i in range(correlations.shape[-1]):
    curve = epoch_mean(correlations[0, :, i], correlations[0, :, i].shape[0]//150)
    plt.plot(curve, alpha=1, label='layer ' + str(i))

plt.title('Layer correlations')
plt.ylabel('Correlation')
plt.xlabel('Epochs')

plt.legend()
plt.show()


# print('t1-base %f, t1-MTL %f' %(metrics[0, 0, -1, 0], metrics[0, -1, -1, 0]))
# print('t2-base %f, t2-MTL %f' %(metrics[0, 0, -1, 1], metrics[0, -1, -1, 1]))


# plt.plot(metrics[0, :, -1, 1], label='test')

# plt.plot(metrics[0, :, -2, 1], label='0')
# plt.plot(metrics[1, :, -2, 1], label='1')


# plt.title('T1:full-net validation accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')


raise SystemError(0)
# plt.plot(metrics[1, :, -2, 0], color='b', linestyle='dashed')
# plt.plot(metrics[1, :, 0, 0], color='orange', linestyle='dashed')
# plt.plot(metrics[1, :, 1, 0], color='g', linestyle='dashed')


# plt.plot(metrics[0, :, -2, 1], label='val')
# plt.plot(metrics[0, :, 0, 1], label='train')
# plt.plot(metrics[0, :, 1, 1], label='ft')

# plt.plot(metrics[1, :, -2, 1], label='val')
# plt.plot(metrics[1, :, 0, 1], label='train')
# plt.plot(metrics[1, :, 1, 1], label='ft')

# plt.plot(metrics[0, :, 0, 0], label='train')
# plt.plot(metrics[4, :, 1, 2], label='ft')
# plt.plot(metrics[4, :, 2, 2], label='8')
# plt.plot(metrics[0, :, -2, 3])
# plt.title('T2:home theater vs closet')
plt.title('T1:rock arch vs shoe shop')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# with open('bincif_metrics.pkl', 'rb') as handle:
#     data = pickle.load(handle)
# metrics = data['train_metrics']
# metrics = np.array(metrics)
# print(metrics[0, -1, :])
# print(len(data['test_t1']))

# for idx_run, run in enumerate(data['test_t1']):
#     if idx_run % 5:
#         plt.plot(run, label=idx_run)
# runs = data['test_t1']

# for run in runs[0:5]:
#     plt.plot(run, color='b', alpha=0.5)

# for run in runs[5:10]:
#     plt.plot(run, color='g', alpha=0.5)

# for run in runs[10:15]:
#     plt.plot(run, color='r', alpha=0.5)

# for run in runs[15:20]:
#     plt.plot(run, color='k', alpha=0.5)

# print('base', runs[0][0])
# tot_corr = 0
# for run in runs[:5]:
#     tot_corr += run[-1]
# print(tot_corr/5)

# tot_nocorr = 0
# for run in runs[5:10]:
#     tot_nocorr += run[-1]
# print(tot_nocorr/5)

# tot_nosoft = 0
# for run in runs[10:15]:
#     tot_nosoft += run[-1]
# print(tot_nosoft/5)

# tot_nosoft = 0
# for run in runs[15:20]:
#     tot_nosoft += run[-1]
# print(tot_nosoft/5)


# print(run[5])
# print(data['runs'][10])
# plt.plot(run[0], label=str(0))
# plt.plot(run[1], label=str(1))
# plt.plot(run[10], label=str(10))
# plt.plot(run[11], label=str(11))
# plt.plot(run[20], label=str(20))
# plt.plot(run[21], label=str(21))
# plt.plot(run[30], label=str(30))
# plt.plot(run[31], label=str(31))
# plt.legend()
# plt.show()

# labels = ['class1 vs class2', 'class2 vs class2', 'class3 vs class3']
# plt.bar(range(len(labels)), [2, 3, -1], width=0.8, bottom=0, align='center')
# plt.xticks(range(len(labels)), labels, rotation=45, horizontalalignment='right')
# plt.tight_layout()
# plt.plot()
# plt.show()

