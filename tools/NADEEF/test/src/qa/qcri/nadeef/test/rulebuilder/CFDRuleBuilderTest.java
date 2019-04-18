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

package qa.qcri.nadeef.test.rulebuilder;

import com.google.common.collect.Lists;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import qa.qcri.nadeef.core.datamodel.Predicate;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.ruleext.CFDRuleBuilder;

import java.util.Collection;
import java.util.List;

/**
 * Test on CFD Rule.
 */
@RunWith(JUnit4.class)
public class CFDRuleBuilderTest {
    @Before
    public void setup() {
        try {
            Bootstrap.start();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    @After
    public void tearDown() {
        Bootstrap.shutdown();
    }

    @Test
    public void parseTest() {
        try {
            List<String> value = Lists.newArrayList(
                "A, B,C , D, E|FF, GG, HH, JJ",
                "a1, _, _, _, E2, f2, _, _, j3",
                "_, b1, _, _, _, f3, _, h1, _"
            );
            CFDRuleBuilder ruleBuilder = new CFDRuleBuilder();
            Collection<Rule> rules =
                ruleBuilder.name("cfd1")
                .table("test1")
                .value(value)
                .build();

            Assert.assertEquals(8, rules.size());

            Assert.assertEquals(5, ruleBuilder.getLhs().size());
            Assert.assertEquals(4, ruleBuilder.getRhs().size());
            Assert.assertEquals(2, ruleBuilder.getFilterExpressions().size());
            List<Predicate> filter = ruleBuilder.getFilterExpressions().get(0);
            Assert.assertEquals(9, filter.size());
            filter = ruleBuilder.getFilterExpressions().get(1);
            Assert.assertEquals(9, filter.size());
        } catch (Exception ignore) {
            ignore.printStackTrace();
            Assert.fail(ignore.getMessage());
        }
    }
}
