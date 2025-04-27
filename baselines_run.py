from KernelUCB import KernelUCB
from LinUCB import Linearucb
from Neural_epsilon import Neural_epsilon
from NeuralTS import NeuralTS
from EAP import NeuralUCBDiag
from NeuralNoExplore import NeuralNoExplore
import argparse
import numpy as np
import sys
import time

from load_data import load_yelp, load_movielen

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baselines')
    parser.add_argument('--dataset', default='movielens', type=str, help='mnist, yelp, movielens, disin')

    parser.add_argument("--method", nargs="+", default=["NeuralUCB"],
                        help='list: ["KernelUCB", "LinUCB", "Neural_epsilon", "NeuralTS", "NeuralUCB", "NeuralNoExplore"]')

    parser.add_argument('--lamdba', default='0.1', type=float, help='Regulization Parameter')
    parser.add_argument('--nu', default='0.001', type=float, help='Exploration Parameter')

    args = parser.parse_args()
    dataset = args.dataset
    arg_lambda = args.lamdba
    arg_nu = args.nu

    print("running data:", args.dataset)
    print("running methods:", args.method)

    for method in args.method:

        regrets_all = []
        for i in range(5):

            start_time = time.time()

            # 数据集要进行替换
            b = load_movielen()
            num_arms = b.n_arm

            print("Number of arms:", num_arms)

            if method == "KernelUCB":
                model = KernelUCB(b.dim, arg_lambda, arg_nu)

            elif method == "LinUCB":
                model = Linearucb(b.dim, arg_lambda, arg_nu)

            elif method == "Neural_epsilon":
                epsilon = 0.01
                model = Neural_epsilon(b.dim, epsilon)

            elif method == "NeuralTS":
                model = NeuralTS(b.dim, b.n_arm, m=100, sigma=arg_lambda, nu=arg_nu)

            elif method == "NeuralUCB":
                model = NeuralUCBDiag(b.dim, lamdba=arg_lambda, nu=arg_nu, hidden=100)



            elif method == "NeuralNoExplore":
                model = NeuralNoExplore(b.dim)
            else:
                print("method is not defined. --help")
                sys.exit()

            regrets = []
            interval_regrets = []
            sum_regret = 0
            print("Round; Regret; Regret/Round")
            for t in range(15000):
                '''Draw input sample'''
                context, rwd = b.step()
                #pool_items, context, rwd, chosen_item_id = b.step()
                arm_select = model.select(context)

                # 输出选择的手臂 ID
                #print(f"Round {t}: Selected arm ID in original pool: {chosen_item_id}")

                reward = rwd[arm_select]



                if method == "LinUCB" or method == "KernelUCB":
                    model.train(context[arm_select], reward)

                elif method == "Neural_epsilon" or method == "NeuralUCB" or method == "NeuralTS" or method == "NeuralNoExplore":
                    #model.update(context[arm_select], reward)

                    model.update(context[arm_select], reward, arm_select)  # neural_1更新模型



                    if t < 1000:
                        if t % 10 == 0:
                            loss = model.train(t)

                    else:
                        if t % 100 == 0:
                            loss = model.train(t)



                regret = np.max(rwd) - reward
                sum_regret += regret
                regrets.append(sum_regret)
                if t % 50 == 0:

                    print('{}: {:}, {:.4f}'.format(t, sum_regret, sum_regret / (t + 1)))

            end_time = time.time()
            elapsed_time = end_time - start_time

            elapsed_time_sec = int(elapsed_time)



            print("run:", i, "; ", "regret:", sum_regret, "; Time taken: {} seconds".format(elapsed_time_sec))

            np.savetxt("./RR/movielens{}_run_{}_time_{}s_{}.txt".format(method, i, elapsed_time_sec, sum_regret), regrets)














