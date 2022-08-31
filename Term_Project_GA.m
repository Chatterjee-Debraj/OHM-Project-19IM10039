% OHM Term Project
% 19IM10039 - Debraj Chatterjee
%% Pure Genetic Algorithm
clc;
clear;
close all;
%% Input Data
D = 60; %Financial Institution Deposit
K = 0.15;  %Required Reserve Ratio
L = [10 25 4 11 18 3 17 15 9 10]; %Loan Sizes
r_L = [0.021 0.022 0.021 0.027 0.025 0.026 0.023 0.021 0.028 0.022];%Interest Rates
Loss = [0.0002 0.0058 0.0001 0.0003 0.0024 0.0002 0.0058 0.0002 0.0001 0.0001]; %Expected Loss
Rating = ['AAA' 'BB' 'A' 'AA' 'BBB' 'AAA' 'BB' 'AAA' 'A' 'A'];
r_T = 0.01; %Customer transaction Rate
r_D = 0.009; %Deposit Rate

Delta = 0.0025; %Institutional cost
Beta = r_D*D;  %Cost of Demand Deposit
T = (1-K)*D - L;  %Institutional Transaction Cost
c = numel(L); %Number of customers

%% GA settings
n = 60; %Population Size
max_gen = 60; %Max number of generations

CP = 0.8; %Crossover Probability
MP = 0.006;  %Mutation Probability

%% Random Population Initialization
tic; %start stopwatch
population = randi([0,1],n,c); %Random 0 or 1 setting for n solutions of c dimensions each
global_best_fitness = 0;
global_best_solution = zeros(1,c);
iter_best_fitness = zeros(n+1,1);   %To store best fitness in each generation
iter_avg_fitness = zeros(n+1,1);    %To store average population fitness in each generation
iter_best_index = zeros(n+1,1); %To store index of best solution in each generation

pop_fitnesses = fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D);%Clculate fitness of each solution in the population
[iter_best_fitness(1,1), iter_best_index(1,1)] = max(fitness(population,r_L, r_T, L, Loss, T, Beta, n, K, D));
global_best_fitness = iter_best_fitness(1,1);
iter_avg_fitness(1,1) = mean(fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D));
global_best_solution = population(iter_best_index(1,1),:);

%% GA algorithm
for iter = 1:max_gen    %Global Loop / stopping criteria
    %SELECTION
    parents_indices = zeros(n,1);   %Array to store indices of selected parents
    pop_fitnesses = fitness(population,r_L, r_T, L, Loss, T, Beta, n, K,D);
    parent_pop = zeros(n,c);%Parent Population
    for parent = 1:n
        selected_index = roulette_wheel(pop_fitnesses); %Selection through weighted roulette wheel
        parent_pop(parent,:) = population(selected_index,:);
    end

    %CROSSOVER
    intermediate_pop = parent_pop;
    for parent_set = 1:n/2 %Each parent set consists of 2 parents
        parent1(1,:) = parent_pop(parent_set*2 - 1,:); %Choose parent 1
        parent2(1,:) = parent_pop(parent_set*2,:); %Choose parent 2

        crossover_random = rand();
        if crossover_random<=CP
            two_children = two_point_crossover(parent1,parent2); %Two children from crossover
            intermediate_pop(parent_set*2-1,:) = two_children(1,:); %Induct children into intermediate population
            intermediate_pop(parent_set*2,:)  = two_children(2,:);
        else 
            intermediate_pop(parent_set*2 - 1,:) = parent1(1,:); %If no crossover, parents go into intermediate population
            intermediate_pop(parent_set*2,:) = parent2(1,:);
        end


    end

    %MUTATION
    for individual = 1:n
        for gene = 1:c
            mutation_num = rand();
            if mutation_num<=MP
                intermediate_pop(individual,gene) = ...
                digit_flip(intermediate_pop(individual,gene)); %Flip digit wherever mutation favorable
            end
        end
    end

    %FITNESS COMPARISON
    intermediate_pop_fitness = fitness(intermediate_pop,r_L, r_T, L, Loss, T, Beta, n, K,D); %Intermediate population fitness
    parent_pop_fitness = fitness(parent_pop, r_L, r_T, L, Loss, T, Beta, n, K,D); %Parent population fitness
    for i = 1:n
        if intermediate_pop_fitness < parent_pop_fitness
        intermediate_pop(i,:) = parent_pop(i,:);
        end
    end
    population = intermediate_pop;
    [iter_best_fitness(iter+1,1),iter_best_index] = max(fitness(population,r_L, r_T, L, Loss, T, Beta, n, K,D));
    iter_avg_fitness(iter+1,1) = mean(fitness(population,r_L, r_T, L, Loss, T, Beta, n, K,D));

    if iter_best_fitness(iter+1,1)>global_best_fitness
        global_best_fitness = iter_best_fitness(iter+1,1);
        global_best_solution = population(iter_best_index,:);
    end
end
toc;

disp(global_best_solution);
disp(global_best_fitness);
disp(iter_avg_fitness(max_gen+1,1));

%% 
subplot(1, 2, 1);
plot(1:max_gen+1, iter_best_fitness);
xlabel('Iterations');
ylabel('Best Fitness of population');
subplot(1, 2, 2);
plot(1:max_gen+1, iter_avg_fitness);
xlabel('Iterations');
ylabel('Average Fitness of population');

%% Helping Functions
% Fitness Calculation
function F = fitness(population, r_L, r_T, L, Loss, T, Beta, pop_size,K,D)   
F = zeros(pop_size,1);
    for i = 1:pop_size
       rev(i) = sum(population(i,:).*(r_L.*L - Loss)); %Revenue from sanctioned loans
       expense(i) = sum(population(i,:).*(r_T.*T));
       F(i) = max(0,rev(i) + expense(i) - Beta - sum(population(i,:).*Loss)...
           - penalty(population(i,:),L,K,D)); %Penalty if constraint violated
    end

end

%PENALTY FUNCTION
function p = penalty(chromosome,L,K,D)
    multiplier  = 0.5;
    p = 0;
    if (sum(chromosome.*L) > (1-K)*D) %If amount loaned out is greater than max allowed limit
        p = sum(chromosome.*L) - (1-K)*D; %apply penalty
        p = multiplier*p;
    end
end

% SELECTION FUNCTION
function selected_parent_index = roulette_wheel(fitnesses)
    n = size(fitnesses,1);
    random_num = rand();
    total_fitness = sum(fitnesses);
    cum_fit = zeros(n,1);
    fitness_ratio = zeros(n,1);
    cum_fit(1,1) = fitnesses(1,1);
    fitness_ratio(1,1) = cum_fit(1,1)/total_fitness; %Weght of roulette wheel
    if random_num<=fitness_ratio(1,1)
        selected_parent_index = 1;
    else 
        for i = 2:n
            cum_fit(i,1) = cum_fit(i-1,1) + fitnesses(i,1);
            fitness_ratio(i,1) = cum_fit(i,1)/total_fitness;
            if fitness_ratio(i-1,1)<random_num && fitness_ratio(i,1)>=random_num %Random number falls in the zone of ith parent
                selected_parent_index = i;
                break;
            end
        end
    end
end

%Two-point crossover function
function two_children = two_point_crossover(parent1, parent2)
    c = size(parent1,2);
    child1 = parent1;
    child2 = parent2;
    point1 = randi(c); %Random point after which crossover starts
    point2 = randi(c); %Random point where crossover ends

    for point = point1+1 : point2
        child1(1,point) = parent2(1,point); %Exchange of genes
        child2(1,point) = parent1(1,point); %Exchange of genes
    end
    two_children = [child1;
                    child2];
end

%MUTATION HELPER FUNCTION
function flipped_digit = digit_flip(digit)
        if digit == 0
            flipped_digit = 1;
        else
            flipped_digit = 0;
        end
    end




