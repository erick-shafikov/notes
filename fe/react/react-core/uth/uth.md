Порядок вызовов внутренних функций React, по стадиям

# Стадия виртуального dom

На этой стадии формируется виртуальный dom

Ниже callstack функций:

- root
- legacyRenderSubtreeContainer (неконкурентный режим)
- - legacyCreateRootFromDOMContainer
- unbatchedUpdates
- - executionContext
- updateContainer
- - requestUpdateLane
- - - lane
- - enqueueUpdate
- - fiber
- - - fiber.child
- - entangleTransition
- scheduleUpdateOnFiber
- - workInProgressRoot
- - fiberNode

# Стадия scheduled renders

Callstack:

- performSyncWorkOnRoot
- - flushPassiveEffects
- renderRootSync
- workLoopSync
  ```js
  // псевдокод
  function workLoop(isYieldy) {
    if (!isYieldy) {
      while (nextUnitOfWork !== null) {
        nextUnitOfWork = performUnitOfWork(nextUnitOfWork);
      }
    } else {...}
  }
  ```
- - performUnitOfWork - получает fiber узел дерева workInProgress
    ```js
    //
    function performUnitOfWork(workInProgress) {
      let next = beginWork(workInProgress);
      if (next === null) {
        next = completeUnitOfWork(workInProgress);
      }
      return next;
    }
    ```
- - workInProgress
- - workLoopSync (цикл)
- - - workInProgress.alternate
- - - workInProgress.tag
- - - workInProgress.child
- - - reconcileChildren
- - - placeSingleChild -> FiberNode

# Стадия commit

Два прохода по дереву работ:

- выполняет все вставки, обновления, удаления и размонтирования DOM (хоста)
- Затем React назначает дерево finishedWork на FiberRoot, помечая дерево workInProgress как current

CallStack:

- commitRoot
- - flushPassiveEffects
- - workInProgressRoot
- commitMutationEffects
- commitMutationEffectsOnFiber
- commitPlacement
- - finishedWork.flags
- - stateNode
- insertOrAppendPlacementNodeIntoContainer
