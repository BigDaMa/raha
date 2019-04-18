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

package qa.qcri.nadeef.test.core;

import com.google.common.collect.Lists;
import org.junit.Assert;
import org.junit.Test;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Fix;
import qa.qcri.nadeef.core.datamodel.Violation;
import qa.qcri.nadeef.core.pipeline.EquivalentClass;
import qa.qcri.nadeef.core.pipeline.FixDecisionMaker;
import qa.qcri.nadeef.test.TestDataRepository;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

/**
 * Test for FixDecisionMakerBase.
 */
public class FixDecisionMakerTest {
    private List<Fix> loadFix(File path) throws IOException {
        List<Fix> result = Lists.newArrayList();
        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line = reader.readLine();
        Fix.Builder fixBuilder = new Fix.Builder().vid(Violation.UnknownId);
        Cell.Builder cellBuilder = new Cell.Builder();
        while ((line = reader.readLine()) != null) {
            String[] tokens = line.split(";");
            Cell leftCell = cellBuilder.tid(Integer.parseInt(tokens[0]))
                .column(tokens[1])
                .value(tokens[2])
                .build();
            Cell rightCell = cellBuilder.tid(Integer.parseInt(tokens[3]))
                .column(tokens[4])
                .value(tokens[5])
                .build();
            Fix fix = fixBuilder.left(leftCell).right(rightCell).build();
            result.add(fix);
        }
        return result;
    }

    @Test
    @SuppressWarnings("unchecked")
    public void test1() {
        try {
            List<Fix> fixes = loadFix(TestDataRepository.getFixTestData1());
            FixDecisionMaker eq = new EquivalentClass(null);
            Collection<Fix> result = eq.decide(fixes);
            Assert.assertEquals(4, result.size());
            for (Fix fix : result) {
                Cell left = fix.getLeft();
                String value = fix.getRightValue();
                switch (left.getTid()) {
                    case 2:
                    case 3:
                    case 4:
                        Assert.assertEquals("a1", value);
                        break;
                    case 6 :
                        Assert.assertEquals("b1", value);
                        break;
                    default:
                        Assert.fail("Incorrect changed tuple.");
                        break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }
}
