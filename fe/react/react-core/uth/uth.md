Порядок вызовов внутренних функций React, по стадиям

- render(element, container, callback)
- legacyRenderSubtreeContainer(parentComponent. children, container) (неконкурентный режим) - вызывает весь цикл рендера в react, далее выполняются основные стадии

# Стадия виртуального dom

На этой стадии формируется виртуальный dom

Глобальные переменные:

- root
- lane
- fiber, свойства:
- - child
- workInProgressRoot
- fiberNode
- executionContext
- entangleTransition

Ниже callstack функций которые вызываются при подготовке:

- legacyCreateRootFromDOMContainer
- - requestUpdateLane
- - enqueueUpdate (fiber, update, lane)

Далее вызываются функции для основного цикла react:

- unbatchedUpdates
- updateContainer(element, container, parentChildren, callback) - рендерит в FiberRoot
- - scheduleUpdateOnFiber(fiber, lane, eventTime) - в функции помечается время
- - - performSyncWorkOnRoot

Далее параллельно вызываются две функции, которые обозначают две стадии работы react

# Стадия scheduled renders

Основная задача посчитать разницу между текущим состоянием и следующим

Callstack функций:

- renderRootSync(root, lanes)
- - prepareFreshStack(root, lanes)
- - flushPassiveEffects

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

Основная задача отобразить дерево компонентов

Два прохода по дереву работ:

- выполняет все вставки, обновления, удаления и размонтирования DOM (хоста)
- Затем React назначает дерево finishedWork на FiberRoot, помечая дерево workInProgress как current

глобальные переменные:

- finishedWork.flags
- stateNode

CallStack:

- commitRoot
- - flushPassiveEffects

- commitMutationEffects
- commitMutationEffectsOnFiber
- commitPlacement
- insertOrAppendPlacementNodeIntoContainer - функция которая вызывает размещение элемента в дереве
