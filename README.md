## CycleGAN的pytorch代码实现
---

### 目录
1. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

## 所需环境
requirements.txt

## 文件下载
权值的百度网盘地址如下：    
链接：https://pan.baidu.com/s/1qn_po_OAsaVf38Obbnau3g 提取码：c7zy  

数据集可以通过百度网盘下载：   
链接：https://pan.baidu.com/s/1qn_po_OAsaVf38Obbnau3g 提取码：c7zy 

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，直接运行test.py，即可生成图片，生成图片位于output文件夹中。    
### b、使用自己训练的权重 
1. 按照训练步骤训练。    
2. 在test.py文件里面，在如下部分修改generator_\*2*使其对应训练好的文件；   
3. default='./saved/ukiyoe2photogauss/G_AB_4.pth'
```python
def test():
    ## 超参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./datasets/ukiyoe2photogauss', help='root directory of the dataset')
    parser.add_argument('--channels', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='./saved/ukiyoe2photogauss/G_AB_4.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='./saved/ukiyoe2photogauss/G_BA_4.pth', help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)
```
3. 运行test.py，即可生成图片，生成图片位于output文件夹中。 

## 训练步骤
1. 训练前将期望生成的图片文件放在datasets文件夹下。
3. 运行train.py文件进行训练，训练过程中生成的图片可查看images文件夹下的图片。  
