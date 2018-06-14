import math
import cmath
import numpy as np


class HawkesOrigin(object):
    """
    HawkesOrigin,
    on the one hand, it defines the APIs of a hawkes process model
    on the other hand, this is also a executable, full-feature Hawkes Process Class.But no performance optimization is
    considered, so the performance is low.
    """
    def __init__(self, data_source, event_count, kernel, iteration,  init_strategy,
                 hyper_parameter, optimize_directly, converge_threshold, time_slot):
        """
        Construct a new Hawkes Model
        :param data_source:
        Input Data Requirement
        Date Structure: {id_index: [(event_index, event_time), (event_index, event_time),...]}
        brace means dictionary, square bracket means list, bracket means tuple. Notations follow the convention of
        document.

        event_index: integer, from 1 to n_j, n_j is the length of a sample sequence, smaller the index, earlier the
        event time of a event
        event_time: integer. Define the event time of first event of a sample as 0. Define the event time of other
        event as the time interval between them to the first event. 'Day' is the recommend unit for our case
        event_id: integer, each id indicates a sample sequence
        the event list, i.e., [(event_index, event_time), (event_index, event_time),...] should be sorted
        according to event time.
        :param event_count: the number of unique event type
        :param kernel: kernel function ,'exp' or 'Fourier'
        :param iteration: if user doesn't set this parameter explicitly, optimization procedure will stop when the
        log-likelihood converges. if we set it, the procedure will stop when log-likelihood converges or the
        optimization is equal to the parameter
        this number
        :param init_strategy:
        :param hyper_parameter: if kernel is exp, we can set omega by dictionary, e.g. {'omega': 2}, the 'default'
        means omega will be set as 1 automatically
        :param optimize_directly: True/False, indicating whether the optimize procedure will be executed
        automatically after initialization procedure accomplished
        :param converge_threshold: log-likelihood converge threshold
        :param time_slot: time slot count
        """
        if hyper_parameter is None:
            self.hyper_parameter = {'omega': 1}
        else:
            self.hyper_parameter = hyper_parameter
        self.data_source = data_source
        self.excite_kernel = kernel
        self.event_count = event_count
        self.iter = iteration
        self.init_strategy = init_strategy
        self.time_slot = time_slot
        self.hyper_parameter = hyper_parameter
        self.auto_optimize = optimize_directly
        self.log_likelihood_tendency = []
        self.base_intensity = self.initialize_base_intensity()
        self.mutual_intensity = self.initialize_mutual_intensity()
        self.auxiliary_variable = self.establish_auxiliary_variable()
        self.event_type_count_list = self.event_number_of_each_slot()
        self.event_sample_count_list = self.event_number_of_each_sample()
        self.converge_threshold = converge_threshold
        print("Hawkes Process Model Initialize Accomplished")

        if optimize_directly:
            print("auto optimize")
            self.optimization(converge_threshold)

    # initialization relevant function
    def initialize_base_intensity(self):
        """
        initialize base intensity according to initial strategy
        default: normal(0, 1)
        :return: base_intensity
        """
        base_intensity = None
        if self.init_strategy == 'default':
            base_intensity = np.random.normal(0, 1, (self.event_count, 1))
        else:
            pass

        if base_intensity is None:
            raise RuntimeError('illegal initial strategy')
        return base_intensity

    def initialize_mutual_intensity(self):
        """
        initialize mutual intensity intensity according to initial strategy
        default: normal(0, 1)
        :return: mutual intensity
        """
        mutual_excite_intensity = None
        if self.init_strategy == 'default':
            mutual_excite_intensity = np.random.normal(0, 1, (self.event_count, self.event_count))
        else:
            pass

        if mutual_excite_intensity is None:
            raise RuntimeError('illegal initial strategy')
        return mutual_excite_intensity

    def establish_auxiliary_variable(self):
        """
        consult eq. 8, 9, 10
        for the i-th event in a sample list, the length of corresponding auxiliary_list is i too.
        The j-th entry of auxiliary_list (j<i) means the probability that the i-th event of the sample sequence is
        'triggered' by the j-th event. The i-th entry of auxiliary_list indicates the probability that the i-th event
        are triggered by base intensity. Obviously, the sum of one list should be one

        :return: auxiliary_map, data structure {id : [auxiliary_map_0, auxiliary_map_1, ...]}
        auxiliary_map_i {list_0, list_1, ...,}
        """
        auxiliary_map = {}
        for item_id in self.data_source:
            auxiliary_event_map = {}
            list_length = len(self.data_source[item_id])
            for i in range(0, list_length):
                single_event_auxiliary_list = []
                for j in range(-1, i):
                    single_event_auxiliary_list.append(0)
                auxiliary_event_map[i] = single_event_auxiliary_list
            auxiliary_map[item_id] = auxiliary_event_map
        return auxiliary_map

    def event_number_of_each_slot(self):
        data_source = self.data_source
        event_list_full = []
        for j in data_source:
            event_list = data_source[j]
            for event in event_list:
                event_list_full.append(event)
        event_list_full = sorted(event_list_full, key=lambda event_time: event_time[1])
        time_unit = (event_list_full[-1]['event_time'] - event_list_full[0]['event_time']) / self.time_slot

        count_list = []
        for i in range(0, self.time_slot):
            count_list.append(0)

        for item in event_list_full:
            slot_index = item['event_time'] // time_unit
            count_list[slot_index] += 1
        return count_list

    def event_number_of_each_sample(self):
        event_number_of_each_sample = {}
        for j in self.data_source:
            count_vector = np.array([self.event_count, 1])
            for event in self.data_source[j]:
                event_index = event['event_index']
                count_vector[event_index][0] += 1
            event_number_of_each_sample[j] = count_vector
        return event_number_of_each_sample

    # optimization
    def optimization(self, converge_threshold):
        # set iteration count
        if self.iter is None:
            iteration_no = 100000
        else:
            iteration_no = self.iter

        for i in range(0, iteration_no):
            # EM Algorithm
            self.expectation_step()
            self.maximization_step()

            # converge test
            current_log_likelihood = self.log_likelihood_calculate()
            previous_log_likelihood = self.log_likelihood_tendency[-1]
            if abs(previous_log_likelihood-current_log_likelihood) < converge_threshold:
                break
            else:
                self.log_likelihood_tendency.append(current_log_likelihood)
            
            if i % 100 == 0:
                print("optimize iteration: "+str(i))
        print("optimization accomplished")

    # expectation step and corresponding function
    def expectation_step(self):
        for j in self.data_source:
            event_list = self.data_source[j]
            list_length = len(event_list)
            for i in range(0, list_length):
                if i == list_length-1:
                    self.auxiliary_variable[j][i][i] = self.calculate_q_ii(j=j, i=i)
                else:
                    for l in range(0, i):
                        self.auxiliary_variable[j][i][l] = self.calculate_q_il(j=j, i=i, _l=l)

    def calculate_q_ii(self, j, i):
        """
        according to eq. 9
        :param j:
        :param i:
        :return:
        """
        event_list = self.data_source[j]
        i_event_index = event_list[i]['event_index']
        i_event_time = event_list[i]['event_time']

        nominator = self.base_intensity[i_event_index][0]

        # calculate denominator
        denominator = 0
        denominator += self.base_intensity[i_event_index][0]
        for l in range(0, i):
            l_event_index = event_list[l]['event_index']
            l_event_time = event_list[l]['event_time']
            alpha = self.mutual_intensity[i_event_index][l_event_index]
            kernel = self.kernel_calculate(early_event_time=l_event_time, late_event_time=i_event_time)
            denominator += alpha * kernel
        q_ii = nominator/denominator
        return q_ii

    def calculate_q_il(self, j, i, _l):
        """
        according to eq. 10
        :param j:
        :param i:
        :param _l: the underline is added to eliminate the ambiguous
        :return:
        """

        event_list = self.data_source[j]
        i_event_index = event_list[i]['event_index']
        i_event_time = event_list[i]['event_time']
        l_event_index = event_list[_l]['event_index']
        l_event_time = event_list[_l]['event_time']
        alpha = self.mutual_intensity[i_event_index][l_event_index]
        kernel = self.kernel_calculate(early_event_time=l_event_time, late_event_time=i_event_time)

        nominator = alpha*kernel

        # calculate denominator
        denominator = 0
        denominator += self.base_intensity[i_event_index][0]
        for l in range(0, i):
            l_event_index = event_list[l]['event_index']
            l_event_time = event_list[l]['event_time']
            alpha = self.mutual_intensity[i_event_index][l_event_index]
            kernel = self.kernel_calculate(early_event_time=l_event_time, late_event_time=i_event_time)
            denominator += alpha * kernel
        q_il = nominator/denominator

        return q_il

    # maximization step and corresponding function
    def maximization_step(self):
        """
        define alpha_(c, c_c) means the intensity that c_c triggers c
        :return:
        """
        for i in self.base_intensity.shape[0]:
            self.update_mu(i)
        for i in self.mutual_intensity.shape[0]:
            for j in self.mutual_intensity.shape[1]:
                self.update_alpha(c=i, c_c=j)

    def update_mu(self, c):
        """
        according to eq. 16
        :param c:
        :return:
        """
        nominator = 0
        denominator = 0
        # calculate nominator
        for j in self.data_source:
            for i in self.data_source[j]:
                event_index = self.data_source[j][i]['event_index']
                if event_index == c:
                    nominator += self.auxiliary_variable[j][i][i]
        # calculate denominator
        for j in self.data_source:
            first_event_time = self.data_source[j][0]['event_time']
            last_event_time = self.data_source[j][-1]['event_time']
            denominator += last_event_time-first_event_time
        mu = nominator/denominator
        self.base_intensity[c][0] = mu

    def update_alpha(self, c, c_c):
        """
        according to eq. 17
        :param c: the event that is affected
        :param c_c: the event that affect c
        :return:
        """
        nominator = 0
        denominator = 0

        # calculate nominator
        for j in self.data_source:
            event_list = self.data_source[j]
            for i in range(1, len(event_list)):
                i_event_index = event_list[i]['event_index']
                for k in range(0, i):
                    k_event_index = event_list[k]['event_index']
                    if c == i_event_index and c_c == k_event_index:
                        nominator += self.auxiliary_variable[j][i][k]
        # calculate denominator
        for j in self.data_source:
            event_list = self.data_source[j]
            last_event_time = event_list[-1]['event_time']
            for l in range(0, self.event_count):
                for k in range(0, len(event_list)):
                    k_event_index = event_list[k]['event_index']
                    k_event_time = event_list[k]['event_time']
                    if c == l and c_c == k_event_index:
                        denominator += self.kernel_integral(last_event_time-k_event_time, 0)

        alpha = nominator/denominator
        self.mutual_intensity[c][c_c] = alpha

    # calculate log-likelihood
    def log_likelihood_calculate(self):
        """
        according to eq. 6
        calculate the log likelihood based on current parameter
        :return:
        """
        log_likelihood = 0

        # calculate part one of log-likelihood
        # according to equation 6
        for j in self.data_source:
            list_length = len(self.data_source[j])

            # part 1
            for i in range(0, list_length):
                part_one = self.part_one_calculate(j=j, i=i)
                log_likelihood += part_one

            # part 2
            for u in range(0, self.event_count):
                part_two = self.part_two_calculate(j=j, u=u)
                log_likelihood -= part_two
        return log_likelihood

    def part_one_calculate(self, j, i):
        """
        according to of eq. 7
        :param j:
        :param i:
        :return:
        """
        part_one = 0

        i_event_index = self.data_source[j][i]['event_index']
        i_event_time = self.data_source[j][i]['event_time']
        mu = self.base_intensity[i_event_index][0]
        part_one += mu

        for l in range(0, i):
            l_event_index = self.data_source[j][l]['event_index']
            l_event_time = self.data_source[j][l]['event_time']

            alpha = self.mutual_intensity[i_event_index][l_event_index]
            kernel = self.kernel_calculate(early_event_time=l_event_time, late_event_time=i_event_time)
            part_one += alpha * kernel

        part_one = math.log(part_one)
        return part_one

    def part_two_calculate(self, u, j):
        """
        according to eq. 12
        :param u:
        :param j:
        :return:
        """
        part_two = 0
        last_event_time = self.data_source[j][-1]['event_time']
        first_event_time = self.data_source[j][0]['event_time']
        part_two += self.base_intensity[u]*(last_event_time-first_event_time)

        for k in range(0, len(self.data_source[j])):
            k_event_index = self.data_source[j][k]['event_index']
            k_event_time = self.data_source[j][k]['event_time']

            lower_bound = 0
            upper_bound = last_event_time-k_event_time
            alpha = self.mutual_intensity[u][k_event_index]

            part_two += alpha*self.kernel_integral(lower_bound=lower_bound, upper_bound=upper_bound)

        return part_two

    # auxiliary function
    def kernel_calculate(self, early_event_time=None, late_event_time=None):
        kernel_type = self.excite_kernel
        if kernel_type == 'default' or kernel_type == 'exp':
            if not self.hyper_parameter.__contains__('omega'):
                raise RuntimeError('illegal hyper_parameter, omega lost')
            omega = self.hyper_parameter['omega']
            kernel_value = math.exp(-1*omega*(late_event_time-early_event_time))
            return kernel_value
        elif kernel_type == 'fourier' or kernel_type == 'Fourier':
            y_omega = []
            for k in range(0, self.time_slot+1):
                omega_k = 2*math.pi/self.time_slot*k
                y_omega_k = 0
                for i in range(0, self.time_slot):
                    y_omega_k += cmath.exp(-1*omega_k*i*complex(0, 1))*self.event_type_count_list[i]
                y_omega.append(y_omega_k)

            k_omega = []
            for k in range(0, self.time_slot+1):
                if k == 0:
                    nominator = 0
                    denominator = 0
                    nominator += y_omega[k]
                    for item in self.base_intensity:
                        nominator -= 2*math.pi*item
                    for j in self.data_source:
                        for c in range(self.event_count):
                            for c_c in range(self.event_count):
                                denominator += self.mutual_intensity[c][c_c]*self.event_sample_count_list[j][c]
                    k_omega.append(nominator/denominator)
                else:
                    nominator = y_omega[k]
                    denominator = 0

                    for j in self.data_source:
                        for item in self.data_source[j]:
                            event_index = item['event_index']
                            event_time = item['event_time']
                            for c in range(self.event_count):
                                for c_c in range(self.event_count):
                                    if event_index == c_c:
                                        alpha = self.mutual_intensity[c][c_c]
                                        omega = 2 * math.pi / self.time_slot * k
                                        exp = cmath.exp(complex(0, 1)*event_time*omega*-1)
                                        denominator += exp*alpha
                    k_omega.append(nominator/denominator)

            # recover
            kappa = 0
            for i in range(0, self.time_slot):
                omega = 2 * math.pi / self.time_slot * i
                kappa += k_omega[i]*cmath.exp(complex(0, 1)*omega*(late_event_time-early_event_time))
            kappa = kappa/self.time_slot
            return kappa
        else:
            raise RuntimeError('illegal kernel name')

    def kernel_integral(self, upper_bound, lower_bound):
        """
        calculate the integral of kernel function
        :param upper_bound:
        :param lower_bound: for the property of hawkes process, lower bound is always 0
        :return:
        """
        if upper_bound < lower_bound:
            raise RuntimeError("upper bound smaller than lower bound")

        kernel_type = self.excite_kernel
        if kernel_type == 'default' or kernel_type == 'exp':
            if not self.hyper_parameter.__contains__('omega'):
                raise RuntimeError('illegal hyper_parameter, omega lost')
            omega = self.hyper_parameter['omega']
            kernel = (math.exp(-1*omega*lower_bound)-math.exp(-1*omega*upper_bound))/omega
            return kernel
        elif kernel_type == 'fourier' or kernel_type == 'Fourier':
            y_omega = []
            for k in range(0, self.time_slot + 1):
                omega_k = 2 * math.pi / self.time_slot * k
                y_omega_k = 0
                for i in range(0, self.time_slot):
                    y_omega_k += cmath.exp(-1 * omega_k * i * complex(0, 1)) * self.event_type_count_list[i]
                y_omega.append(y_omega_k)

            k_omega = []
            for k in range(0, self.time_slot + 1):
                if k == 0:
                    nominator = 0
                    denominator = 0
                    nominator += y_omega[k]
                    for item in self.base_intensity:
                        nominator -= 2 * math.pi * item
                    for j in self.data_source:
                        for c in range(self.event_count):
                            for c_c in range(self.event_count):
                                denominator += self.mutual_intensity[c][c_c] * \
                                               self.event_sample_count_list[j][c]
                    k_omega.append(nominator / denominator)
                else:
                    nominator = y_omega[k]
                    denominator = 0

                    for j in self.data_source:
                        for item in self.data_source[j]:
                            event_index = item['event_index']
                            event_time = item['event_time']
                            for c in range(self.event_count):
                                for c_c in range(self.event_count):
                                    if event_index == c_c:
                                        alpha = self.mutual_intensity[c][c_c]
                                        omega = 2 * math.pi / self.time_slot * k
                                        exp = cmath.exp(complex(0, 1) * event_time * omega * -1)
                                        denominator += exp * alpha
                    k_omega.append(nominator / denominator)
            integral = 0
            for k in range(0, self.time_slot):
                omega = 2 * math.pi / self.time_slot * k
                integral += k_omega[k]*complex(0, 1)/omega*(1-cmath.exp(complex(0, 1)*omega*(upper_bound-lower_bound)))
            integral = integral/self.time_slot
            return integral
        else:
            raise RuntimeError('illegal kernel name')

    # output essential variable
    def get_log_likelihood(self):
        return self.log_likelihood_tendency

    def get_mutual_intensity(self):
        return self.mutual_intensity

    def get_base_intensity(self):
        return self.base_intensity

    def get_auxiliary_variable(self):
        return self.auxiliary_variable


class Hawkes(HawkesOrigin):
    """
    the function of this class is as same as HawkesOrigin while this class optimize the performance by reconstruct
    some code and vectorization
    """
    def __init__(self, data_source, event_count, kernel="exp", iteration=None, init_strategy="default",
                 hyper_parameter=None, optimize_directly=False, converge_threshold=0.000001, time_slot=100):
        HawkesOrigin.__init__(self, data_source, event_count, kernel, iteration, init_strategy, hyper_parameter,
                              optimize_directly, converge_threshold, time_slot)
        self.auxiliary_variable_denominator = None
        self.mu_update_nominator_vector = None
        self.mu_update_denominator_vector = None
        self.alpha_update_denominator_matrix = None
        self.alpha_update_nominator_matrix = None

    def alpha_nominator_update(self):
        alpha_matrix = np.zeros([self.event_count, self.event_count])
        data_source = self.data_source
        for j in data_source:
            event_list = data_source[j]
            for i in range(0, len(event_list)):
                for k in range(0, i):
                    i_event_index = event_list[i]['event_index']
                    k_event_index = event_list[k]['event_index']
                    alpha_matrix[i_event_index][k_event_index] += self.auxiliary_variable[i][j][k]
        self.alpha_update_nominator_matrix = alpha_matrix

    def alpha_denominator_update(self):
        alpha_matrix = np.zeros([self.event_count, self.event_count])
        for j in self.data_source:
            event_list = self.data_source[j]
            last_event_time = event_list[-1]['event_time']
            for l in range(0, self.event_count):
                for k in range(0, len(event_list)):
                    k_event_index = event_list[k]['event_index']
                    k_event_time = event_list[k]['event_time']
                    alpha_matrix[l][k_event_index] = self.kernel_integral(last_event_time-k_event_time, 0)
        self.alpha_update_denominator_matrix = alpha_matrix

    def mu_nominator_update(self):
        nominator = np.zeros([self.event_count, 1])
        for j in self.data_source:
            event_list = self.data_source[j]
            for i in range(0, len(event_list)):
                i_event_index = event_list[i]['event_index']
                nominator[i_event_index][0] += self.auxiliary_variable[j][i][i]
        self.mu_update_nominator_vector = nominator

    def mu_denominator_update(self):
        data_source = self.data_source
        denominator = 0
        for j in data_source:
            event_list = data_source[j]
            first_time = event_list[0]['event_time']
            last_time = event_list[-1]['event_time']
            denominator += last_time-first_time
        self.mu_update_denominator_vector = denominator

    def auxiliary_variable_denominator_update(self):
        denominator_map = {}
        data_source = self.data_source
        for j in data_source:
            single_denominator_map = {}
            event_list = data_source[j]
            for i in range(len(event_list)):
                i_event_index = event_list[i]['event_index']
                i_event_time = event_list[i]['event_time']

                denominator = 0
                denominator += self.base_intensity[i_event_index][0]
                for l in range(0, i):
                    l_event_index = event_list[l]['event_index']
                    l_event_time = event_list[l]['event_time']
                    alpha = self.mutual_intensity[i_event_index][l_event_index]
                    kernel = self.kernel_calculate(early_event_time=l_event_time, late_event_time=i_event_time)
                    denominator += alpha * kernel
                single_denominator_map[i] = denominator
            denominator_map[j] = single_denominator_map

        self.auxiliary_variable_denominator_cache = denominator_map

    def calculate_q_ii(self, j, i):
        """
        according to eq. 9
        :param j:
        :param i:
        :return:
        """
        event_list = self.data_source[j]
        i_event_index = event_list[i]['event_index']
        nominator = self.base_intensity[i_event_index][0]
        denominator = self.auxiliary_variable_denominator_cache[j][i]
        q_ii = nominator/denominator
        return q_ii

    def calculate_q_il(self, j, i, _l):
        """
        according to eq. 10
        :param j:
        :param i:
        :param _l: the underline is added to eliminate the ambiguous
        :return:
        """

        event_list = self.data_source[j]
        i_event_index = event_list[i]['event_index']
        i_event_time = event_list[i]['event_time']
        l_event_index = event_list[_l]['event_index']
        l_event_time = event_list[_l]['event_time']
        alpha = self.mutual_intensity[i_event_index][l_event_index]
        kernel = self.kernel_calculate(early_event_time=l_event_time, late_event_time=i_event_time)

        nominator = alpha*kernel
        denominator = self.auxiliary_variable_denominator_cache[j][i]
        return nominator/denominator

    def maximization_step(self):
        """
        :return:
        """
        self.alpha_nominator_update()
        self.alpha_denominator_update()
        self.mu_nominator_update()
        self.mu_denominator_update()
        self.mutual_intensity = self.alpha_update_nominator_matrix / self.alpha_update_denominator_matrix
        self.base_intensity = self.mu_update_nominator_vector/self.mu_update_denominator_vector

    def expectation_step(self):
        # Update denominator of auxiliary variable
        self.auxiliary_variable_denominator_update()
        for j in self.data_source:
            event_list = self.data_source[j]
            list_length = len(event_list)
            for i in range(0, list_length):
                if i == list_length-1:
                    self.auxiliary_variable[j][i][i] = self.calculate_q_ii(j=j, i=i)
                else:
                    for l in range(0, i):
                        self.auxiliary_variable[j][i][l] = self.calculate_q_il(j=j, i=i, _l=l)


if __name__ == "_main_":
    print('Hello World')
