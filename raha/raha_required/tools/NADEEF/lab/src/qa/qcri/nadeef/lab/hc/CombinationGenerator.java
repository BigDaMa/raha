/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.lab.hc;

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

/**
 * Combination generator is used to generate different set of predicates to fix in order to
 * reach cardinality minimality fix. The detail concept can be found in the paper of
 * Holistic Cleaning.
 */
public class CombinationGenerator<T> {
    private List<T> inputList;
    private int[] state;
    private int num;
    public CombinationGenerator(Collection<T> inputList) {
        Preconditions.checkNotNull(inputList != null && inputList.size() > 0);
        this.inputList = new ArrayList<>(inputList);
        state = new int[this.inputList.size()];
        state[0] = this.inputList.size();
        num = 1;
    }

    /**
     * Gets the next combination. This is a lazy sequence.
     */
    public HashSet<T> getNext() {
        if (!search(0)) {
            if (num == inputList.size())
                return null;
            else {
                num ++;
                reset(0);
            }
        }

        HashSet<T> output = Sets.newHashSet();
        for (int i = 0; i < num; i ++)
            output.add(inputList.get(state[i]));
        return output;
    }

    private void reset(int from) {
        for (int i = from; i < num; i ++)
            state[i] = state.length - (num - i);
    }

    private boolean search(int level) {
        if (level >= num)
            return false;

        int cur = state[level];
        if (search(level + 1))
            return true;

        if (cur == 0 || (level > 0 && (cur - state[level - 1] == 1)))
            return false;

        state[level] --;
        reset(level + 1);
        return true;
    }
}
