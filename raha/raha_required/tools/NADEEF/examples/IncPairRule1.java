import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import qa.qcri.nadeef.core.datamodel.*;

import java.util.Collection;
import java.util.List;

public class IncPairRule1 extends PairTupleRule {
    @Override
    public void initialize(String id, List<String> tableNames) {
        super.initialize(id, tableNames);
    }

    @Override
    public Collection<Table> block(Collection<Table> tables) {
        Table table = Iterables.get(tables, 0);
        return table.groupOn("C");
    }

    /**
     * Detect method.
     * @param tuplePair tuple pair.
     * @return violation set.
     */
    @Override
    public Collection<Violation> detect(TuplePair tuplePair) {

        Tuple left = tuplePair.getLeft();
        Tuple right = tuplePair.getRight();
        Violation violation = new Violation(getRuleName());
        violation.addTuple(left);
        violation.addTuple(right);
        return Lists.newArrayList(violation);
    }

    /**
     * Repair of this rule.
     *
     * @param violation violation input.
     * @return a candidate fix.
     */
    @Override
    public Collection<Fix> repair(Violation violation) {
        return null;
    }
}
