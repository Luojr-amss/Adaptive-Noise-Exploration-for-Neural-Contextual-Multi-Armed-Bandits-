
from EAP import EAP
import argparse
import numpy as np
import sys
import time

from load_data import load_yelp, load_movielen

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baselines')
    parser.add_argument('--dataset', default='movielens', type=str, help='mnist, yelp, movielens, disin')

    parser.add_argument("--method", nargs="+", default=["NeuralUCB"],
                        help='')

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

            if method == "NeuralUCB":
                model = EAP(b.dim, lamdba=arg_lambda, nu=arg_nu, hidden=100)




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



                if method == "NeuralUCB" :
                    model.update(context[arm_select], reward, arm_select)



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














