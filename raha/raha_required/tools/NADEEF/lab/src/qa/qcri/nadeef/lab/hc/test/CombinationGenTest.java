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

package qa.qcri.nadeef.lab.hc.test;

import org.junit.Test;
import qa.qcri.nadeef.lab.hc.CombinationGenerator;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class CombinationGenTest {
    @Test
    public void simpleTest() {
        List<Integer> input = new ArrayList<>();
        input.add(1);
        input.add(2);
        input.add(3);
        input.add(4);

        CombinationGenerator<Integer> combinator =
            new CombinationGenerator<>(input);

        HashSet<Integer> combination = combinator.getNext();
        int pre = 0;
        while (combination != null) {
            int size = combination.size();
            // Assert.assertTrue(size >= pre);
            if (size > pre)
                pre = size;

            for (Integer v : combination)
                System.out.print(v + " ");
            System.out.println();
            combination = combinator.getNext();
        }
    }
}
