import pickle
import tensorflow as tf
import time
from common import get_args, experiment_setup
from tqdm import tqdm

load = False

"""
Code by James Li
https://github.com/hakrrr/I-HGG

Minor modifications by Timo Friedl
"""


def train():
    # Getting arguments from command line + defaults
    # Set up learning environment including, gym env, ddpg agent, hgg/normal learner, tester
    args = get_args(clear_log=not load)
    env, env_test, agent, buffer, learner, tester = experiment_setup(args)

    if load:
        play_dir = "log/100-FetchPick2AndPlace-v1-hgg-ce-BAK2/"
        meta_path = "{}saved_policy-best.meta".format(play_dir)
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(agent.sess, tf.train.latest_checkpoint(play_dir))

    args.logger.summary_init(agent.graph, agent.sess)

    # Progress info
    args.logger.add_item('Epoch')
    args.logger.add_item('Cycle')
    args.logger.add_item('Episodes@green')
    args.logger.add_item('Timesteps')
    args.logger.add_item('TimeCost(sec)')

    # Algorithm info
    for key in agent.train_info.keys():
        args.logger.add_item(key, 'scalar')

    # Test info
    for key in tester.info:
        args.logger.add_item(key, 'scalar')

    args.logger.summary_setup()
    counter = 0

    best_success = -1

    for epoch_cycle in tqdm(range(args.epoches * args.cycles)):
        epoch = epoch_cycle // args.cycles
        cycle = epoch_cycle % args.cycles

        args.logger.tabular_clear()
        args.logger.summary_clear()
        start_time = time.time()

        # Learn
        goal_list = learner.learn(args, env, env_test, agent, buffer, write_goals=args.show_goals)

        # Log learning progresss
        tester.cycle_summary()
        args.logger.add_record('Epoch', str(epoch) + '/' + str(args.epoches))
        args.logger.add_record('Cycle', str(cycle) + '/' + str(args.cycles))
        args.logger.add_record('Episodes', buffer.counter)
        args.logger.add_record('Timesteps', buffer.steps_counter)
        args.logger.add_record('TimeCost(sec)', time.time() - start_time)

        # Save learning progress to progress0.csv file
        args.logger.save_csv()

        args.logger.tabular_show(args.tag)
        args.logger.summary_show(buffer.counter)

        # Save latest policy
        policy_file = args.logger.my_log_dir + "saved_policy-latest"
        agent.saver.save(agent.sess, policy_file)

        # Save policy if new best_success was reached
        if args.logger.values["Success"] > best_success:
            best_success = args.logger.values["Success"]
            policy_file = args.logger.my_log_dir + "saved_policy-best"
            agent.saver.save(agent.sess, policy_file)
            args.logger.info("Saved as best policy to {}!".format(args.logger.my_log_dir))

        if cycle == args.cycles - 1:
            # Save periodic policy every epoch
            policy_file = args.logger.my_log_dir + "saved_policy"
            agent.saver.save(agent.sess, policy_file, global_step=epoch)
            args.logger.info("Saved periodic policy to {}!".format(args.logger.my_log_dir))

            # Plot current goal distribution for visualization (G-HGG only)
            if args.learn == 'hgg' and goal_list and args.show_goals != 0:
                name = "{}goals_{}".format(args.logger.my_log_dir, epoch)
                if args.graph:
                    learner.sampler.graph.plot_graph(goals=goal_list, save_path=name)
                with open('{}.pkl'.format(name), 'wb') as file:
                    pickle.dump(goal_list, file)

            tester.epoch_summary()

    tester.final_summary()


if __name__ == '__main__':
    try:
        train()
    except RuntimeError as err:
        msg = str(err)
        if 'Failed to initialize OpenGL' in msg:
            print('{0}. Please try to execute "unset LD_PRELOAD" before running training.'
                  .format(msg))
            exit(0)
        else:
            raise err
