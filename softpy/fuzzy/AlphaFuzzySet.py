# Ghiotto_Alessandro_513944_laboratory_project.py

# + --------------------------------------------------------------- +
# |  Laboratory Project - Fuzzy Systems and Evolutionary Computing  |
# |                                                                 |
# |                  Ghiotto Alessandro 513944                      |
# + --------------------------------------------------------------- +


from .fuzzyset import ContinuousFuzzySet
from .operations import ContinuousFuzzyCombination
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


class AlphaFuzzySet(ContinuousFuzzySet):
    '''
    Abstract representation of fuzzy sets in the horizontal representation,
    a fuzzy set is represented in terms of the corresponding collection of alpha-cuts.
    We are considering alpha-cuts in which the extremes are included (not strong alpha-cuts).

    ATTRIBUTES:
        - collection : dictionary whose keys are the alpha values and the values are the corresponding (list of) crisp sets
        - min        : minimum value of the support of the fuzzy set
        - max        : maximum value of the support of the fuzzy set
    '''
    collection : dict
    min        : np.number
    max        : np.number

    def __init__(self, collection: dict):
        '''
        INPUTS:
            - collection : dictionary whose keys are the alpha values and the values are the corresponding crisp sets
                keys   : float in [0,1]
                values : list of lists with length 2, representing the interval [a,b] of the alpha-cut.
                         it supports also a single interval, in this case it is converted to a list of lists.
                            example: [a, b] -> [[a, b]]
                         it supports also a single value, in this case it is converted to a list [a, b], with a=b
                            example: [a] -> [[a, a]]

                example: { 0.1 : [0, 10], 0.5 : [[0, 5], [7.5, 10]], 1 : [0]}
                            1   -
                            0.5 ------   ---
                            0.1 ------------

        STEPS:
            - check for the type of the input
            - check that respect the properties of the fuzzy sets
            - sort (ASCENDING ORDER) the intervals in the alpha-cuts
            - sort (ASCENDING ORDER) the alpha-cuts in the collection
            - remove redundant alpha-cuts
        '''
        # NOTATION:
        # collection = {alpha1 : cut1, alpha2 : cut2, ...}
        # cut = [interval1, interval2, ...]
        # interval = [a, b]

        # Here we do most of the work, once we have checked the input, and we have regularized it
        # we can store it in the attributes and use it in the other methods

        if type(collection) != dict:
            raise TypeError("The collection of alpha-cuts should be a dictionary, is %s" % type(collection))

        # check the type and format of the alpha-cuts
        # and also I regularize them, so that I can iterate over them in the same way
        # for example if I have a single interval, I convert it to a list of lists
        # if I have a single value, I convert it to a list [a, b], with a=b
        for alpha, cut in collection.items():
            if not np.issubdtype(type(alpha), np.number):
                raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
            if alpha < 0 or alpha > 1:
                raise ValueError("Alpha should be in [0,1], is %s" % alpha)
            if alpha == 0:
                raise ValueError("Alpha=0 is not allowed, the corresponding cut covers the whole real line for any fuzzy set")
            if type(cut) != list:
                raise TypeError("Alpha-cut should be a list. For alpha=%s received %s" % (alpha, type(cut)))
            
            # if I have a single interval, I convert it to a list of lists
            if len(cut) == 1 and np.issubdtype(type(cut[0]), np.number):
                cut = [[cut[0], cut[0]]] # [a] -> [[a, a]]
                collection[alpha] = cut
            elif np.issubdtype(type(cut[0]), np.number) and np.issubdtype(type(cut[1]), np.number) and len(cut) == 2:
                cut = [cut] # [a, b] -> [[a, b]]
                collection[alpha] = cut

            # now I check the single intervals in a cut
            for interval in cut:
                if type(interval) != list:
                    raise TypeError("Alpha-cut should be a list of lists. For alpha=%s received %s" % (alpha, type(interval)))
                # if I have a single value, I convert it to a list [a, b], with a=b
                if len(interval) == 1:
                    cut[cut.index(interval)] = [interval[0], interval[0]]
                    interval = [interval[0], interval[0]]
                if len(interval) != 2:
                    raise TypeError("Alpha-cut should be a list containing lists of 2 elements. For alpha=%s received %s" % (alpha, interval))
                if not np.issubdtype(type(interval[0]), np.number) or not np.issubdtype(type(interval[1]), np.number):
                    raise TypeError("Intervals in the alpha-cut should be [a,b] with a,b floats. For alpha=%s received %s" % (alpha, interval))
                if interval[0] > interval[1]:
                    raise ValueError("Intervals in the alpha-cut should be [a,b] with a <= b. For alpha=%s received %s" % (alpha, interval))

            # REORDER THE INTERVALS IN THE ALPHA-CUT
            # the first appearing in the list is the first that we see if we go from
            # left to right in the real line
            cut.sort(key=lambda x: x[0]) 

            # I check that the intervals are not overlapping
            if len(cut) > 1:
                for i in range(len(cut)-1):
                    if cut[i][1] >= cut[i+1][0]: # I can just check the extremes (one by one) since I have ordered the intervals
                        # We could have easily merged them, but I prefer to raise an error (it could be just a mistake in the input)
                        raise ValueError("The intervals in the alpha-cut should not overlap, "
                                         "specify better your alpha-cut. For alpha=%s received %s" % (alpha, cut))
       

        # ORDER THE COLLECTION BY ALPHA VALUES (ASCENDING ORDER)
        # collection.keys[0]  : the lowest alpha value, the corresponding cut is the support of the fuzzy set -> [min, max]
        # collection.keys[-1] : the highest alpha value, the correspoding cut is contained in all the others
        collection = dict(sorted(collection.items()))

        # Utility function to check if all the intervals in cut1 are contained in the intervals of cut2
        # we use this function to check if the alpha-cuts are nested (putted here for readability)
        def is_cut1_in_cut2(cut1, cut2):
            for interval1 in cut1:
                contained = False
                for interval2 in cut2:
                    # if the interval1 is contained in interval2
                    if interval2[0] <= interval1[0] and interval1[1] <= interval2[1]:
                        # this is ok, is contained in one of the intervals of cut2
                        contained = True
                        break
                # if the interval1 is not contained in any of the intervals of cut2
                if not contained:
                    return False
            return True
       
        # IMPORTANT: if [b,c] belongs to the a-cut, then it also belongs to all a'-cuts, for a' < a.
        # CHECK THAT THE ALPHA-CUTS ARE NESTED
        previous_cut = list(collection.values())[0]
        previous_alpha = list(collection.keys())[0]
        for alpha, cut in collection.items():
            if not is_cut1_in_cut2(cut, previous_cut):
                raise ValueError("The alpha-cuts should be nested, "
                                  "for alpha=%s the cut is not contained in the cut of alpha=%s" % (alpha, previous_alpha))
            previous_cut = cut
            previous_alpha = alpha

        # REMOVE REDUNDANT ALPHA-CUTS
        # if two alpha-cuts are equal, we keep the one with the greatest alpha,
        # the smaller one doesn't add any information
        # I remove one alpha-cut at a time, until are all different
        alpha_to_be_removed = True
        while alpha_to_be_removed:
            alpha_to_be_removed = None
            previous_cut = None
            previous_alpha = None
            for alpha, cut in collection.items():
                if previous_cut == cut: # if two successive alpha-cuts are equal
                    # I will remove the smaller one (which is the previous one, since are ordered by alpha ascending)
                    alpha_to_be_removed = previous_alpha 
                    break # I remove one at a time
                previous_cut = cut
                previous_alpha = alpha

            if alpha_to_be_removed is not None:
                del collection[alpha_to_be_removed]
            # if alpha_to_be_removed is None, there are no two successive alpha-cuts that are equal, so I stop the loop
        
        #-------------------------
        # ATTRIBUTES
        self.collection = collection
        self.min = list(collection.values())[0][ 0][ 0] # left bound of the first interval of the first alpha-cut
        self.max = list(collection.values())[0][-1][-1] # right bound of the last interval of the first alpha-cut

    
    def __call__(self, arg: np.number) -> np.number:
        """
        Returns the membership degree of the given argument in the fuzzy set.
        """
        if not np.issubdtype(type(arg), np.number):
            raise TypeError("Arg should be float, is %s" % type(arg))
        
        # our items are stored in ASCENDING ORDER of alpha
        # we iterate and we return the last alpha (the biggest) for which the argument is contained
        # in the corresponding cut
        prev_alpha = 0.0
        is_contained = False
        for alpha, cut in self.collection.items():
            # check if the argument is contained in the cut
            for interval in cut:
                if interval[0] <= arg <= interval[1]:
                    is_contained = True
                    break

            # if the argument is not contained in the cut, we return the previous alpha
            # which is the last alpha for which the argument is contained in the corresponding cut
            if not is_contained:
                return prev_alpha
            
            # if the argument is contained in the cut, we update the previous alpha for the next iteration
            prev_alpha = alpha
            is_contained = False

        return prev_alpha # if the argument is contained in the last cut, we return the last alpha

        
    def __getitem__(self, alpha: np.number) -> list:
        """
        returns the interval of the alpha-cut corresponding to the given alpha value.
        """
        if not np.issubdtype(type(alpha), np.number):
            raise TypeError("Alpha should be a float in [0,1], is %s" % type(alpha))
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be in [0,1], is %s" % alpha)
        
        if alpha == 0:
            return -np.inf, np.inf
        
        # alpha-cuts are stored in ASCENDING ORDER of alpha
        # we iterate and we return the first cut for which the current_alpha is greater or equal to the given alpha
        for current_alpha, cut in self.collection.items():
            if current_alpha >= alpha:
                return cut
    

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlphaFuzzySet):
            return NotImplemented
        
        # if the corresponding collections are the same, the two fuzzy sets are equal
        return self.collection == other.collection
    

    def update_collection(self):
        """
        If the collection is modified, we can check if it is still a valid collection of alpha-cuts.
        And we update the min and max attributes.
        """
        self.__init__(self.collection)


    def plot(self, vline_x=None, hline_y=None, ax=None, title=None, linewidth=4, markersize=8):
        """
        Plots the alpha-cuts.
        vline_x -> show the membership degree of a specific element
        hline_y -> show the alpha-cut of a specific alpha

        INPUTS:
            - vline_x: float, plot a vertical line at the specified x value.
            - hline_y: float, plot a horizontal line at the specified y value.
            - ax: matplotlib.axes.Axes, the axis object where to plot the alpha-cuts. If None, a new plot is created.
            - title: str, the title of the plot. If None, a default title is used.
            - linewidth: float, the width of the lines representing the intervals in the alpha-cuts.
            - markersize: float, the size of the markers representing the intervals in the alpha-cuts.

        RETURNS:
            matplotlib.axes.Axes: The axis object with the plotted alpha-cuts.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        for alpha, cut in self.collection.items():
            for interval in cut:
                # we draw each interval at y=alpha (for each interval in any alpha-cut)
                ax.plot(interval, [alpha, alpha], marker='D', linewidth=linewidth, markersize=markersize)  # Draw horizontal line for any interval
        
        # Plot the vertical line if vline_x is specified
        if vline_x is not None:
            vline = ax.axvline(x=vline_x, color='black', linestyle='--', linewidth=2, label='x=%s' % vline_x)
        
        # Plot the horizontal line if hline_y is specified
        if hline_y is not None:
            if hline_y == 0:
                ax.axhline(y=hline_y, color='black', linestyle='--', linewidth=2)
            else:
                for interval in self[hline_y]:
                    ax.plot(interval, [hline_y, hline_y], color='black', linestyle='--', marker='D', linewidth=2, markersize=markersize)
            hline = mlines.Line2D([], [], color='black', marker='D', markersize=5, label='alpha=%s' % hline_y) # handle for the legend

        # Set plot labels, title and axis limits
        ax.set_xlabel('x-axis')
        ax.set_ylabel('alpha')
        title = title if title is not None else 'Plot of the Alpha-Cuts'
        ax.set_title(title)
        ax.grid(True)
        x_margin = (self.max - self.min)/10
        ax.set_xlim(self.min - x_margin, self.max + x_margin)
        ax.set_ylim(-0.1, 1.1)
        
        # Add legend to the plot
        if vline_x is not None and hline_y is not None:
            ax.legend(handles=[vline, hline])
        if vline_x is not None and hline_y is None:
            ax.legend(handles=[vline])
        if vline_x is None and hline_y is not None:
            ax.legend(handles=[hline])

        return ax
    
    
    def fuzziness(self):
        pass

    def hartley(self):
        pass



########################################################################################################


class AlphaFuzzyCombination(AlphaFuzzySet, ContinuousFuzzyCombination):
    '''
    Implements a binary operator on AlphaFuzzySet instances

    RETURNS: a new AlphaFuzzySet instance representing the combination of the two input fuzzy sets
    It is exactly like an AlphaFuzzySet, but it stores also as attributes the two input fuzzy sets and the operator used

    NOTE: for any a in [0,1], [f'(F1, F2)]a = f([F1]a, [F2]a)

    ATTRIBUTES:
        - left: AlphaFuzzySet,
        - right: AlphaFuzzySet,
        - op: function from R^2 to R, which is assumed to be continuous and associative
        - collection: horizontal representation of the fuzzy set resulting from the combination of the two input fuzzy sets
        - min: minimum value of the support of the fuzzy set
        - max: maximum value of the support of the fuzzy set
    '''
    left       : AlphaFuzzySet
    right      : AlphaFuzzySet
    op         : callable
    collection : dict
    min        : np.number
    max        : np.number

    def __init__(self, left: AlphaFuzzySet, right: AlphaFuzzySet, op=None):
        """
        INPUTS:
            - left: AlphaFuzzySet,
            - right: AlphaFuzzySet,
            - op: a function from R^2 to R, which is assumed to be continuous and associative

        STEPS (for computing the resulting collection):
            - take the union of the alpha values of the two input fuzzy sets
            - for each alpha in the union, compute the combination of the two alpha-cuts
            - I apply the operator to the extremes of each pair of intervals in the two alpha-cuts
            - once I have all the possible combinations, I merge the overlapping intervals
            - I store the resulting alpha-cut in the attribute 'collection'

        RETURNS:
            a new AlphaFuzzySet instance representing the combination of the two input fuzzy sets
        """
        if not isinstance(left, AlphaFuzzySet) or not isinstance(right, AlphaFuzzySet):
            raise TypeError("All arguments should be AlphaAuzzySets")
        if op is None:
            raise ValueError("The parameter 'op' should be specified")
        
        # utility function for merging overlapping intervals in a cut (cut = list of intervals)
        def merge_intervals(cut) -> list:
            if not cut:
                return []
            cut.sort(key=lambda x: x[0]) # Sort intervals by their start points (ASCENDING ORDER)
            merged = [cut[0]] # Initialize the merged list with the first interval
            for current_interval in cut[1:]:
                last_merged = merged[-1]
                # If the current interval overlaps with the last merged interval, merge them
                if current_interval[0] <= last_merged[1]:
                    last_merged[1] = max(last_merged[1], current_interval[1])
                else:
                    # If they don't overlap, add the current interval to merged
                    merged.append(current_interval)
            return merged
        
        # WE USE ASSOCIAIVITY OF THE FUNCTION
        # ex f([a, b], [[c, d], [e, f]]) = [f([a, b], [c, d]), f([a, b], [e, f])]
        collection = {}
        all_alphas = set(left.collection.keys()).union(set(right.collection.keys()))
        for alpha in all_alphas: # we iterate over all the alphas that appear in the two collections
            cut1 = left[alpha]
            cut2 = right[alpha]
            cut_result = []
            # we unpack the intervals and we compute the result of the operator on the extremes
            # all possible combination between the intervals of the two AlphaFuzzySets
            for interval1 in cut1:
                for interval2 in cut2:
                    # we consider all combinations, since what will be the min or max depends on the operator
                    # all the '4 angles' of the rectangle given by the two intervals
                    values_at_the_extremes = [op(interval1[0], interval2[0]), 
                                              op(interval1[0], interval2[1]), 
                                              op(interval1[1], interval2[0]), 
                                              op(interval1[1], interval2[1])]
                    resulting_interval = [min(values_at_the_extremes), max(values_at_the_extremes)]
                    cut_result.append(resulting_interval)
            # we have done all the possible combinations, now we have to merge the overlapping intervals
            cut_result = merge_intervals(cut_result)
            collection[alpha] = cut_result
        

        # regularize the collection (sort the intervals in the cuts, remove redundant alpha-cuts)
        # and store it in the attribute 'collection'
        # we also store the min and max attributes
        super().__init__(collection=collection)  # we call the __init__ of AlphaFuzzySet
        
        #-------------------------
        # ATTRIBUTES
        self.left = left
        self.right = right
        self.op = op
        # self.collection = collection
        # self.min = list(collection.values())[0][ 0][ 0]
        # self.max = list(collection.values())[0][-1][-1]

        # once we have the collection attribute, we can inherit everything from AlphaFuzzySet
        
